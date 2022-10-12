import torch
import torch.distributions as dists
from typing import Any, Dict, List, Tuple

from bayes_pcn.const import Kernel
from bayes_pcn.pcnet.util import is_PD


def kernel_log_prob(X_train: torch.Tensor, Y_train: torch.Tensor,
                    X_test: torch.Tensor, Y_test: torch.Tensor,
                    kernel: Kernel, kernel_params: Dict[str, Any],
                    Sigma_trtr_inv: List[torch.Tensor]) -> torch.Tensor:
    """Compute the log probability of the test datapoints Y_test given X_test.

    Args:
        X_train (torch.Tensor): Training feature matrix of shape <n_train x x_dim>.
        Y_train (torch.Tensor): Training label vector of shape <n_train x y_dim>.
        X_test (torch.Tensor): Test matrix of shape <n_test x x_dim>.
        Y_test (torch.Tensor): Test matrix of shape <n_test x y_dim>.
        kernel (Kernel): Kernel type (ex. Kernel.RBF).
        lengthscale (float): The lower this is, the less correlated datapoints are.
        Sigma_trtr_inv (List[torch.Tensor]): A list of training data precision matrices.
            The list's length should be <y_dim> and its members' shapes <n_train x n_train>.

    Returns:
        torch.Tensor: Log probability scalar tensor.
    """
    if X_train is None or Y_train is None:
        # HACK: When no data is in the model, just return the prior over the activations
        Sigma_prior, _, _ = get_kernel_sigmas(X_train=X_test, X_test=None,
                                              kernel=kernel, kernel_params=kernel_params)
        error_mean = torch.zeros(len(Y_test)).to(Y_test.device)
        dist = dists.MultivariateNormal(error_mean, Sigma_prior)
        return dist.log_prob(Y_test.T).sum().unsqueeze(0)

    mu_posterior, Sigma_posterior = kernel_posterior_params(
                                        X_train=X_train, Y_train=Y_train, X_test=X_test,
                                        kernel=kernel, kernel_params=kernel_params,
                                        Sigma_trtr_inv=Sigma_trtr_inv)
    error_mean = torch.zeros(len(Y_test)).to(Y_test.device)
    dist = dists.MultivariateNormal(error_mean, Sigma_posterior)
    return dist.log_prob(Y_test.T - mu_posterior.T).sum().unsqueeze(0)


def kernel_posterior_params(X_train: torch.Tensor, Y_train: torch.Tensor, X_test: torch.Tensor,
                            kernel: Kernel, kernel_params: Dict[str, Any],
                            Sigma_trtr_inv: List[torch.Tensor]
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi output GP forward operation. Assumes independence between output dimensions.
    Assumes that X_train/Y_train exists. Currently only support independent outputs.

    Args:
        X_train (torch.Tensor): Training feature matrix of shape <n_train x x_dim>.
        y_train (torch.Tensor): Training label vector of shape <n_test x 1>.
        X_test (torch.Tensor): Test matrix of shape <n_test x x_dim>.
        kernel (Kernel): Kernel type (ex. Kernel.RBF).
        kernel_params (Dict[str, Any]): Kernel hyperparameter dictionary.
        Sigma_trtr_inv (List[torch.Tensor]): A list of training data precision matrices.
            The list's length should be <y_dim> and its members' shapes <n_train x n_train>.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Predicted label matrix of shape <n_test x y_dim>
            and its covariance tensor of shape <n_test x n_test>.
    """
    mu_train, mu_test = 0., 0.
    _, Sigma_trte, Sigma_tete = get_kernel_sigmas(X_train=X_train, X_test=X_test,
                                                  kernel=kernel, kernel_params=kernel_params)
    mu_posterior = mu_test + Sigma_trte.T.matmul(Sigma_trtr_inv).matmul(Y_train - mu_train)
    Sigma_posterior = Sigma_tete - Sigma_trte.T.matmul(Sigma_trtr_inv).matmul(Sigma_trte)
    if not is_PD(Sigma_posterior):
        print(Sigma_trte)
        Sigma_posterior = torch.eye(len(Sigma_posterior)).to(Sigma_posterior.device) * 1e-6
    return mu_posterior, Sigma_posterior


def get_kernel_sigmas(X_train: torch.Tensor, X_test: torch.Tensor,
                      kernel: Kernel, kernel_params: Dict[str, Any]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kernel == Kernel.RBF:
        return rbf_sigmas(X_train=X_train, X_test=X_test, **kernel_params)
    elif kernel == Kernel.RELU:
        return relu_sigmas(X_train=X_train, X_test=X_test, **kernel_params)
    else:
        raise NotImplementedError()


def rbf_sigmas(X_train: torch.Tensor, X_test: torch.Tensor, lengthscale: float, Sigma_obs: float,
               **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_train = X_train.shape[0]
    X_trtr_dot = X_train.matmul(X_train.T)
    X_trtr_norm = X_trtr_dot.diag().repeat(n_train, 1)
    Sigma_trtr = torch.exp(-(X_trtr_norm+X_trtr_norm.T-2*X_trtr_dot)/(2*lengthscale**2))
    Sigma_trtr = Sigma_trtr + Sigma_obs * torch.eye(n_train).to(Sigma_trtr.device)
    if X_test is None:
        return Sigma_trtr, None, None

    n_test = X_test.shape[0]
    X_trte_dot = X_train.matmul(X_test.T)
    X_tete_dot = X_test.matmul(X_test.T)
    X_tete_norm = X_tete_dot.diag().repeat(n_test, 1)
    X_trte_norm_tr = X_trtr_dot.diag().repeat(n_test, 1)
    X_trte_norm_te = X_tete_dot.diag().repeat(n_train, 1)
    Sigma_trte = torch.exp(-(X_trte_norm_tr.T+X_trte_norm_te-2*X_trte_dot)/(2*lengthscale**2))
    Sigma_tete = torch.exp(-(X_tete_norm+X_tete_norm.T-2*X_tete_dot)/(2*lengthscale**2))
    Sigma_tete = Sigma_tete + Sigma_obs * torch.eye(n_test).to(Sigma_tete.device)
    return Sigma_trtr, Sigma_trte, Sigma_tete


def relu_sigmas(X_train: torch.Tensor, X_test: torch.Tensor, Sigma_prior: float, Sigma_obs: float,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_train = X_train.shape[0]
    Phi_train = torch.nn.functional.relu(X_train)
    Sigma_trtr = Phi_train.matmul(Phi_train.T) * Sigma_prior
    Sigma_trtr = Sigma_trtr + Sigma_obs * torch.eye(n_train).to(Sigma_trtr.device)
    if X_test is None:
        return Sigma_trtr, None, None

    n_test = X_test.shape[0]
    Phi_test = torch.nn.functional.relu(X_test)
    Sigma_trte = Phi_train.matmul(Phi_test.T) * Sigma_prior
    Sigma_tete = Phi_test.matmul(Phi_test.T) * Sigma_prior
    Sigma_tete = Sigma_tete + Sigma_obs * torch.eye(n_test).to(Sigma_tete.device)
    return Sigma_trtr, Sigma_trte, Sigma_tete
