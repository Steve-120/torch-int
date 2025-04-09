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
    """
    def __init__(self):
        pass

    @staticmethod
    def layernorm(x: torch.Tensor, weight, bias, dtype: torch.dtype) -> IBertQuant:
        """
        IBERT-style LayerNorm using PyTorch (not CUTLASS).
        https://github.com/kssteven418/I-BERT/blob/1b09c759d6aeb71312df9c6ef74fa268a87c934e/fairseq/quantization/utils/quant_modules.py#L454
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

            # "it converges within at most **four** iterations for any INT32" - I-BERT
            # NOTE: this would be while True loop, but for safety, it is capped at 100 iterations
            # TODO: if the 4 iterations for int8 is true, could we just express this entire thing as
            #       a finite multiplication? it wouldn't hurt to go over 4 iterations
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
        new_scaling_factor = S  # TODO: this is different on L488. what does L488 mean? why are we doing this?

        # --- output
        new_bias = bias.data.detach() / weight.data.detach()
        new_bias_int = torch.div(new_bias, S, rounding_mode="floor").to(dtype)

        new_q_int = y_int + new_bias_int
        new_scaling_factor = S * weight

        return IBertQuant(new_q_int, new_scaling_factor, dtype)


    @staticmethod
    def softmax(quant: IBertQuant) -> IBertQuant:
        """
        IBERT-style SoftMax using PyTorch (not CUTLASS).
        """
        raise NotImplementedError()

    @staticmethod
    def relu(quant: IBertQuant) -> IBertQuant:
        """
        IBERT-style ReLU using PyTorch (not CUTLASS).
        """
        raise NotImplementedError()