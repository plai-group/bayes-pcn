import torch
import torch.nn.functional as F


def local_wta(X_in: torch.Tensor, block_size: int, hard: bool = True) -> torch.Tensor:
    d_batch, d_orig = X_in.shape
    assert d_orig % block_size == 0
    num_blocks = d_orig // block_size
    X_in = X_in.reshape(d_batch, num_blocks, block_size)
    if hard:
        mask_matrix = F.one_hot(torch.argmax(X_in, dim=-1), num_classes=block_size)
        return (X_in * mask_matrix).reshape(d_batch, d_orig)
    else:
        beta = 8.
        return F.softmax(X_in * beta, dim=-1).reshape(d_batch, d_orig)


def dpfp(X_in: torch.Tensor, nu: int = 1) -> torch.Tensor:
    """Projects the input to a higher dimensional space, promoting sparsity and orthogonality.
    Refer to https://arxiv.org/pdf/2102.11174.pdf.

    Args:
        X_in (torch.Tensor): Tensor of shape <d_batch x d_x>
        nu (int, optional): Capacity controlling hyperparameter. Defaults to 1.

    Returns:
        torch.Tensor: Tensor of size <d_batch x (2 * d_x * nu)>
    """
    x = torch.cat([F.relu(X_in), F.relu(-X_in)], dim=-1)
    # x = torch.cat([F.elu(X_in)+1, F.elu(-X_in)+1], dim=-1)
    x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, nu+1)], dim=-1)
    x_repeat = torch.cat([x] * nu, dim=-1)
    return x_repeat * x_rolled
