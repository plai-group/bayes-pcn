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


def gmm_log_joint(X: torch.Tensor, pi: torch.Tensor, sigma_obs: float,
                  R: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
    """Calculate the log joint over the data matrix.

    Args:
        X (torch.Tensor): Data matrix of shape <d_batch x d_features>.
        pi (torch.Tensor): Component probability vector of shape <n_components>.
        sigma_obs (float): Observation standard deviation.
        R (torch.Tensor): Mean matrix of shape <n_components x d_features>.
        U (torch.Tensor, optional): Diagonal covariance matrix of shape
                                    <n_components x n_components>.
                                    Should only be provided if using Bayesian GMM.

    Returns:
        torch.Tensor: Log joint matrix of shape <d_batch x n_components>.
    """
    n_components, (d_batch, d_features) = len(pi), X.shape
    X = X.T.repeat(n_components, 1, 1)    # <n_components x d_features x d_batch>
    log_pi = pi.unsqueeze(-1).repeat(1, d_batch).log()  # <n_components x d_batch>
    R = R.unsqueeze(-1)                   # <n_components x d_features x 1>
    component_sigmas = torch.ones_like(R) * sigma_obs
    if U is not None:
        prior_Sigmas = U.diag().unsqueeze(-1).repeat(1, d_features).unsqueeze(-1)
        component_sigmas = (component_sigmas**2 + prior_Sigmas)**0.5
    component_log_probs = dists.Normal(loc=R, scale=component_sigmas).log_prob(X)
    component_log_probs = component_log_probs.sum(dim=1)  # <n_components x d_batch>
    return (log_pi + component_log_probs).T


def gmm_log_marginal(X: torch.Tensor, pi: torch.Tensor, sigma_obs: float,
                     R: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
    """Calculate the marginal log likelihood of the data matrix.

    Args:
        X (torch.Tensor): Data matrix of shape <d_batch x d_features>.
        pi (torch.Tensor): Component probability vector of shape <n_components>.
        sigma_obs (float): Observation standard deviation.
        R (torch.Tensor): Mean matrix of shape <n_components x d_features>.
        U (torch.Tensor, optional): Diagonal covariance matrix of shape
                                    <n_components x n_components>.
                                    Should only be provided if using Bayesian GMM.

    Returns:
        torch.Tensor: Log marginal vector of shape <d_batch>.
    """
    log_joint = gmm_log_joint(X=X, pi=pi, sigma_obs=sigma_obs, R=R, U=U)
    return torch.logsumexp(log_joint, dim=-1)


def gmm_log_posterior(X: torch.Tensor, pi: torch.Tensor, sigma_obs: float,
                      R: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
    """Calculate the log posterior over the component indices.

    Args:
        X (torch.Tensor): Data matrix of shape <d_batch x d_features>.
        pi (torch.Tensor): Component probability vector of shape <n_components>.
        sigma_obs (float): Observation standard deviation.
        R (torch.Tensor): Mean matrix of shape <n_components x d_features>.
        U (torch.Tensor, optional): Diagonal covariance matrix of shape
                                    <n_components x n_components>.
                                    Should only be provided if using Bayesian GMM.

    Returns:
        torch.Tensor: Log posterior matrix of shape <d_batch x n_components>.
    """
    log_joint = gmm_log_joint(X=X, pi=pi, sigma_obs=sigma_obs, R=R, U=U)
    log_marginal = torch.logsumexp(log_joint, dim=-1, keepdim=True)
    return log_joint - log_marginal
