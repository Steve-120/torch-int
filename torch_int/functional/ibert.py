import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class IBertQuant:
    """
    All IBERT-style quantization functions.
    """
    q: torch.Tensor
    S: torch.Tensor  # scaling factor as a 0D float tensor
    dtype: torch.dtype

    @classmethod
    def quant(cls, x: torch.Tensor, dtype: torch.dtype):
        """
        Quantize a tensor to a given integer dtype.

        :param x: original, unquantized tensor
        :param dtype: output dtype
        :return: quantized dtype tensor
        """
        # assume alpha (clipping factor) to be the bounds of dtype
        dtype_info = torch.iinfo(dtype)
        alpha = torch.tensor(dtype_info.max, dtype=dtype)
        bits = torch.tensor(dtype_info.bits, dtype=dtype)
        S = alpha / (2**(bits - 1)-1)

        q = torch.clamp(x, -(alpha-1), alpha).div(S).to(dtype)
        return cls(q, S, dtype)

    @classmethod
    def quant_to_sint8(cls, x: torch.Tensor):
        return cls.quant(x, torch.int8)

    @classmethod
    def quant_to_sint32(cls, x: torch.Tensor):
        return cls.quant(x, torch.int32)

    def dequant(self) -> torch.Tensor:
        return self.q * self.S


class IBertComputation:
    """
    Using :py:class:`IBertQuant` to perform non-linear operations.
    This assumes that all weights and bias were already converted to the quantized dtype.
    """
    def __init__(self):
        pass

    @staticmethod
    def layernorm(x: torch.Tensor, weight, bias, dtype: torch.dtype) -> IBertQuant:
        """
        IBERT-style LayerNorm using PyTorch (not CUTLASS). Works for all int dtypes.
        https://github.com/kssteven418/I-BERT/blob/1b09c759d6aeb71312df9c6ef74fa268a87c934e/fairseq/quantization/utils/quant_modules.py#L454

        :param x: original, unquantized tensor
        :param weight: weights for this layer
        :param bias: bias for this layer
        :param dtype: output dtype
        :return: quantized dtype tensor with layernorm applied
        """
        quant = IBertQuant.quant(x, dtype)
        q, S = quant.q, quant.S
        dtype_info = torch.iinfo(dtype)
        dmax, dmin, bits = dtype_info.max, dtype_info.min, dtype_info.bits

        # --- compute mean
        mean_int = torch.div(q.to(torch.int32).sum(), q.numel(), rounding_mode="floor").clamp(dmin, dmax).to(dtype)

        # --- compute var(std)
        numerator = q - mean_int
        numerator_squared = torch.square(numerator)
        var_int = torch.sum(numerator_squared, dim=2, keepdim=True)
        std_int = torch.Tensor(0)

        # square root
        if var_int != 0:
            std_int = torch.bitwise_left_shift(1, torch.floor(bits/2))  # torch.Tensor(2 ** torch.floor(bits/2))

            # "it converges within at most **four** iterations for any **INT32**" - I-BERT
            # NOTE: this would be while True loop, but for safety, it is capped at 100 iterations
            # TODO: if the 4 iterations for int8 is true, could we just express this entire thing as
            #       a finite equation? it wouldn't hurt to go over 4 iterations
            for _ in range(100):
                new_std_int = torch.bitwise_right_shift(
                    torch.floor(std_int + torch.div(var_int, std_int, rounding_mode="floor"))
                , 1)  # / 2

                if new_std_int >= std_int:
                    break  # success!
                std_int = new_std_int
            else:
                raise RuntimeError("LayerNorm failed to converge within 100 iterations")

        # now compute var(std)
        y_int = torch.div(numerator, std_int, rounding_mode="floor").to(dtype)
        # new_scaling_factor = S  # TODO: this is different on L488. what does L488 mean? why are we doing this?

        # --- output
        new_bias = bias.data.detach() / weight.data.detach()
        new_bias_int = torch.div(new_bias, S, rounding_mode="floor").to(dtype)

        new_q_int = y_int + new_bias_int
        new_scaling_factor = S * weight

        return IBertQuant(new_q_int, new_scaling_factor, dtype)

    @staticmethod
    def softmax(quant: IBertQuant) -> IBertQuant:
        """
        IBERT-style SoftMax using PyTorch (not CUTLASS). Works for all int dtypes.
        https://github.com/kssteven418/I-BERT/blob/1b09c759d6aeb71312df9c6ef74fa268a87c934e/fairseq/quantization/utils/quant_modules.py#L578

        :param quant: input quantized tensor
        :return: quantized tensor with softmax applied
        """
        q, S, dtype = quant.q, quant.S, quant.dtype
        q_tilde = q - torch.max(q, dim=-1, keepdim=True).values
        res_exp = IBertComputation._exp(IBertQuant(q_tilde, S, dtype))

        return IBertQuant(
            res_exp.q / torch.sum(res_exp.q, dim=2, keepdim=True),
            res_exp.S,
            dtype
        )

    @staticmethod
    def gelu(quant: IBertQuant) -> IBertQuant:
        """
        IBERT-style GeLU using PyTorch (not CUTLASS). Works for all int dtypes.

        :param quant: input quantized tensor
        :return: quantized tensor with GeLU applied
        """
        q, S, dtype = quant.q, quant.S, quant.dtype
        quant_erf = IBertComputation._erf(IBertQuant(q, torch.div(S, torch.sqrt(2)), dtype))
        q_1 = torch.div(1, quant_erf.S, rounding_mode="floor").to(dtype)
        return IBertQuant(
            q * (quant_erf.q + q_1),
            torch.bitwise_right_shift(S*quant_erf.S, 1),  # / 2
            dtype
        )

    @staticmethod
    def _poly(quant: IBertQuant, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> IBertQuant:
        q, S, dtype = quant.q, quant.S, quant.dtype
        q_b = torch.div(b, S, rounding_mode="floor").to(dtype)
        q_c = torch.div(c, a*torch.square(S), rounding_mode="floor").to(dtype)
        return IBertQuant(
            torch.square(q+q_b) + q_c,
            torch.floor(a*torch.square(S)),
            dtype
        )

    @staticmethod
    def _exp(quant: IBertQuant) -> IBertQuant:
        q, S, dtype = quant.q, quant.S, quant.dtype
        a, b, c = torch.Tensor([0.3585]), torch.Tensor([1.353]), torch.Tensor([0.344])
        ln2_approx = torch.Tensor([0.6931])
        large_n = torch.Tensor([30])

        # make sure we don't get div by 0
        l2_int = torch.div(ln2_approx, S, rounding_mode="floor").to(dtype)
        new_q = torch.max(q, large_n * l2_int)
        z = torch.div(-q, q_ln2, rounding_mode="floor").to(dtype)
        q_p = q + z*q_ln2

        new_quant = IBertComputation._poly(IBertQuant(q_p, S, dtype), a, b, c)
        return IBertQuant(
            torch.bitwise_right_shift(new_quant.q, z),
            new_quant.S,
            dtype
        )

    @staticmethod
    def _erf(quant: IBertQuant) -> IBertQuant:
        q, S, dtype = quant.q, quant.S, quant.dtype
        a, b, c = torch.Tensor(-0.2888), torch.Tensor(-1.769), torch.Tensor(1)
        q_sign = torch.sign(q)
        q_clipped = torch.clamp(torch.linalg.vector_norm(q, dim=2, keepdim=True), max=torch.div(-b, S))

        new_quant = IBertComputation._poly(IBertQuant(q_clipped, S, dtype), a, b, c)
        return IBertQuant(
            q_sign*new_quant.q,
            new_quant.S,
            dtype
        )