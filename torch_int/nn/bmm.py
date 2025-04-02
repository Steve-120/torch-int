import torch
# from .._CUDA import bmm_s8t_s8n_s8t, bmm_s8t_s8n_s32t, bmm_s8t_s8n_f32t


def check_matrices(A: torch.Tensor, B: torch.Tensor, cuda_mode=False) -> None:
    if cuda_mode:
        raise NotImplementedError("CUDA has been disabled for this module until further testing.")

    [batch_size, m, k] = A.shape  # k = lda
    [_, n, ldb] = B.shape

    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise ValueError("Inputs must be int8 numpy arrays")
    if k != ldb:
        raise ValueError("A and B are not compatible.")


class BMM_S8T_S8N_S8T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, A: torch.Tensor, B: torch.Tensor, cuda_mode=False) -> torch.Tensor:
        """
        Batch matmul with 8-bit signed integers where...\n
        - A is transposed
        - B is not transposed
        - C is transposed

        :param A: [B, M, K] int8
        :param B: [B, N, K] int8
        :param cuda_mode: whether to use CUDA kernel
        :return: [B, M, N] int8
        """
        check_matrices(A, B, cuda_mode)
        alpha = self.alpha.item()

        C = torch.matmul(A.to(torch.int32), torch.permute(B.to(torch.int32), (0, 2, 1)))  # accumulation
        return torch.clamp(C * alpha, min=-128, max=127).to(torch.int8)  # quantize

    @staticmethod
    def from_scale(a_scale, b_scale, output_scale):
        bmm_module = BMM_S8T_S8N_S8T(1.0)
        alpha = a_scale * b_scale / output_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_F32T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer('a', torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, A: torch.Tensor, B: torch.Tensor, cuda_mode=False) -> torch.Tensor:
        """
        Batch matmul with 8-bit signed integers where...\n
        - A is transposed
        - B is not transposed
        - C is transposed

        :param A: [B, M, K] int8
        :param B: [B, N, K] int8
        :param cuda_mode: whether to use CUDA kernel
        :return: [B, M, N] float32
        """
        check_matrices(A, B, cuda_mode)
        alpha = self.alpha.item()

        C = torch.matmul(A.to(torch.int32), torch.permute(B.to(torch.int32), (0, 2, 1)))  # accumulation
        return (C * alpha).to(torch.float32)


    @staticmethod
    def from_scale(a_scale, b_scale):
        bmm_module = BMM_S8T_S8N_F32T(1.0)
        alpha = a_scale * b_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_S32T(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, A: torch.Tensor, B: torch.Tensor, cuda_mode=False) -> torch.Tensor:
        """
        Batch matmul with 8-bit signed integers where...\n
        - A is transposed
        - B is not transposed
        - C is transposed

        :param A: [B, M, K] int8
        :param B: [B, N, K] int8
        :param cuda_mode: whether to use CUDA kernel
        :return: [B, M, N] int32
        """
        check_matrices(A, B, cuda_mode)
        alpha = self.alpha.item()

        C = torch.matmul(A.to(torch.int32), torch.permute(B.to(torch.int32), (0, 2, 1)))  # accumulation
        return C * alpha
