import numpy as np
from numpy import linalg as la
import torch
import torch.distributions as dists


def is_PD(A: torch.Tensor):
    if A.numel() > 0:
        # If I check only one of these conditions PyTorch can complain
        pd_check = dists.constraints._PositiveDefinite().check(A)
        cholesky_check = torch.linalg.cholesky_ex(A.flip(-2, -1)).info == 0
        return pd_check and cholesky_check
    return True


def nearest_PD(A: torch.Tensor):
    """Find the nearest positive-definite matrix to input

    Source: https://stackoverflow.com/a/43244194
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_PD(A3):
        return A3.to(A.dtype)

    spacing = np.spacing(la.norm(A))
    eye = np.eye(A.shape[0])
    k = 1
    while not is_PD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += eye * (-mineig * k**2 + spacing)
        k += 1

    return A3.to(A.dtype)


def fixed_indices_exists(fixed_indices: torch.Tensor) -> bool:
    return fixed_indices is not None and fixed_indices.max().item() == 1


def normalize(X_in: torch.Tensor) -> torch.Tensor:
    return X_in / torch.norm(X_in, p=2, dim=-1, keepdim=True)
