from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from numpy import linalg as la
import torch
import torch.distributions as dists
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, NamedTuple, Tuple

from bayes_pcn.const import Kernel
from .a_group import ActivationGroup


class DataBatch(NamedTuple):
    train: Tuple[torch.Tensor, torch.Tensor]
    tests: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    original_shape: torch.Size
    train_pred: Tuple[torch.Tensor, torch.Tensor] = None
    tests_pred: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None
    info: Dict[str, Any] = None


class Prediction(NamedTuple):
    data: torch.Tensor
    a_group: ActivationGroup
    info: Dict[str, Any] = None


class Sample(NamedTuple):
    data: torch.Tensor
    log_joint: float
    info: Dict[str, Any] = None


class UpdateResult(NamedTuple):
    pcnets: List[Any]
    log_weights: torch.Tensor
    info: Dict[str, Any] = None


class LogProbResult(NamedTuple):
    log_prob: torch.Tensor
    layer_log_probs: List[torch.Tensor]


class BaseDistribution(ABC):
    @abstractmethod
    def sample(self) -> ActivationGroup:
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        raise NotImplementedError()


class Dirac(BaseDistribution):
    def __init__(self, mean_vectors: torch.Tensor, X_obs: torch.Tensor, dims: List[int]) -> None:
        self.mean_vectors = mean_vectors.clone()
        self.X_obs = X_obs.clone()
        self.dims = dims
        self.device = X_obs.device

    def sample(self) -> ActivationGroup:
        activations = torch.cat((self.X_obs, self.mean_vectors), dim=-1)
        a_group = ActivationGroup.from_concatenated(activations=activations, dims=self.dims)
        a_group.device = self.device
        return a_group

    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        return torch.zeros(a_group.d_batch).to(self.device)


class DiagMVN(BaseDistribution):
    def __init__(self, mean_vectors: torch.Tensor, stdev_vectors: torch.Tensor,
                 X_obs: torch.Tensor, dims: List[int]) -> None:
        self.mean_vectors = mean_vectors.clone()
        self.stdev_vectors = stdev_vectors.clone()
        self.X_obs = X_obs.clone()
        self.dims = dims
        self.device = X_obs.device

    def sample(self) -> ActivationGroup:
        h_activations = self.mean_vectors + self.stdev_vectors*torch.randn_like(self.stdev_vectors)
        activations = torch.cat((self.X_obs, h_activations), dim=-1)
        a_group = ActivationGroup.from_concatenated(activations=activations, dims=self.dims)
        a_group.device = self.device
        return a_group

    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        data = a_group.get_data(flatten=True, no_obs=True)
        log_prob = dists.Normal(loc=self.mean_vectors, scale=self.stdev_vectors).log_prob(data)
        return log_prob.sum(dim=-1).to(self.device)


def safe_mvn(mean_vectors: torch.Tensor, precision_matrices: torch.Tensor
             ) -> dists.MultivariateNormal:
    try:
        return dists.MultivariateNormal(loc=mean_vectors, precision_matrix=precision_matrices)
    except ValueError:
        safe_precision_matrices = []
        for precision_matrix in precision_matrices:
            if not is_PD(A=precision_matrix):
                precision_matrix = nearest_PD(A=precision_matrix)
            safe_precision_matrices.append(precision_matrix)
        safe_precision_matrices = torch.stack(safe_precision_matrices, dim=0)
        return dists.MultivariateNormal(loc=mean_vectors, precision_matrix=safe_precision_matrices)


class MVN(BaseDistribution):
    def __init__(self, mean_vectors: torch.Tensor, precision_matrices: torch.Tensor,
                 X_obs: torch.Tensor, dims: List[int]) -> None:
        self.dist = safe_mvn(mean_vectors=mean_vectors, precision_matrices=precision_matrices)
        self.X_obs = X_obs.clone()
        self.dims = dims
        self.device = X_obs.device

    def sample(self) -> ActivationGroup:
        sample = self.dist.sample()
        activations = torch.cat((self.X_obs, sample), dim=-1)
        a_group = ActivationGroup.from_concatenated(activations=activations, dims=self.dims)
        a_group.device = self.device
        return a_group

    def log_prob(self, a_group: ActivationGroup) -> torch.Tensor:
        data = a_group.get_data(flatten=True, no_obs=True)
        return self.dist.log_prob(data)


def estimate_free_energy(log_joint_fn: Callable[[ActivationGroup], LogProbResult],
                         a_group: ActivationGroup, n_particles: int = 1) -> LogProbResult:
    """Returns the free energy vector associated with the input ActivationGroup.
    If a_group.stochastic is true, assume a normal variational distribution.
    Otherwise, assume a Dirac delta variational distribution.

    Args:
        log_joint_fn (Callable[[ActivationGroup], torch.Tensor]): A function that accepts
            an ActivationGroup object and returns log probability vector of shape <d_batch>.
        a_group (ActivationGroup): Initial coordinate at the activation space.
        n_particles (int, optional): How many particles to use for approximating the free energy.
            Defaults to 1.

    Returns:
        LogProbResult: An object containing each datapoint's total and layerwise free energy.
    """
    if not a_group.stochastic:
        lp_result = log_joint_fn(a_group)
        return LogProbResult(log_prob=-lp_result.log_prob,
                             layer_log_probs=[-lp for lp in lp_result.layer_log_probs])

    energy = 0.
    for _ in range(n_particles):
        activations = [a_group.data[0]]
        for layer_acts, layer_stdevs in zip(a_group.data[1:], a_group.stdevs):
            layer_acts_p = layer_acts + layer_stdevs * torch.randn_like(layer_stdevs)
            activations.append(layer_acts_p)
        a_group_p = ActivationGroup(activations=activations, no_param=True, stochastic=False)
        lp_result = log_joint_fn(a_group_p)
        energy -= lp_result.log_prob / n_particles

    # batch_entropies = [0.] * len(activations[0])
    # layer_entropies = [0.] * len(a_group.stdevs)
    # for i_layer, layer_stdevs in enumerate(a_group.stdevs):
    #     for i_batch in range(len(batch_entropies)):
    #         entropy_li = 0.5*torch.logdet((2*torch.pi*torch.e*layer_stdevs[i_batch]).diag())
    #         batch_entropies[i_batch] += entropy_li
    #         layer_entropies[i_layer] += entropy_li
    # entropy = torch.tensor(batch_entropies).to(energy.device)

    # elbo = entropy - energy
    # layer_log_joints = torch.tensor([-lp for lp in lp_result.layer_log_probs])
    # layer_entropies = torch.tensor(layer_entropies)
    # layer_elbos = layer_entropies - layer_log_joints
    # return LogProbResult(log_prob=-elbo, layer_log_probs=layer_elbos.tolist())
    batch_entropies = [0.] * len(activations[0])
    for layer_stdevs in a_group.stdevs:
        for i_batch in range(len(batch_entropies)):
            entropy_li = 0.5*torch.logdet((2*torch.pi*torch.e*layer_stdevs[i_batch]).diag())
            batch_entropies[i_batch] += entropy_li
    entropy = torch.tensor(batch_entropies).to(energy.device)

    elbo = entropy - energy
    layer_log_joints = torch.tensor([-lp for lp in lp_result.layer_log_probs])
    return LogProbResult(log_prob=-elbo, layer_log_probs=layer_log_joints.tolist())

    # energy = 0.
    # for _ in range(n_particles):
    #     activations = [a_group.data[0]]
    #     for layer_acts, layer_stdevs in zip(a_group.data[1:], a_group.stdevs):
    #         layer_acts_p = layer_acts + layer_stdevs * torch.randn_like(layer_stdevs)
    #         activations.append(layer_acts_p)
    #     a_group_p = ActivationGroup(activations=activations, no_param=True, stochastic=False)
    #     lp_result = log_joint_fn(a_group_p)
    #     energy -= lp_result.log_prob / n_particles

    # entropy = [0.] * len(activations[0])
    # for layer_stdevs in a_group.stdevs:
    #     for i_batch in range(len(entropy)):
    #         entropy_li = 0.5*torch.logdet((2*torch.pi*torch.e*layer_stdevs[i_batch]).diag())
    #         entropy[i_batch] = entropy[i_batch] + entropy_li
    # entropy = torch.tensor(entropy).to(energy.device)
    # elbo = entropy - energy
    # return -elbo


def maximize_log_joint(log_joint_fn: Callable[[ActivationGroup], LogProbResult],
                       a_group: ActivationGroup, infer_T: int, infer_lr: float,
                       activation_optim: str, fixed_indices: torch.Tensor = None,
                       n_particles: int = 1, **kwargs) -> Dict[str, List[float]]:
    """Move in the space of activation vectors to minimize log joint under the model.
    a_group is modified in place. To clarify, the model is defined by its log_joint_fn.
    Depending on what part of a_group is 'clamped' or not updated by gradient descent,
    this function can be used to only update hidden layer neurons, observation neurons,
    or both. Clamps all layers on method exit.

    NOTE: Autoassociative and heteroassociative recall cannot occur in the same batch.

    Args:
        log_joint_fn (Callable[[ActivationGroup], torch.Tensor]): A function that accepts
            an ActivationGroup object and returns log probability vector of shape <d_batch>.
        a_group (ActivationGroup): Initial coordinate at the activation space.
        infer_T (int): Maximum number of gradient descent iterations.
        infer_lr (float): Gradient descent learning rate.
        activation_optim (str): Which optimizer to use for gradient descent.
        fixed_indices (torch.Tensor, optional): Boolean matrix of shape <d_batch x d_out> that
            denotes which observation neuron indices to prevent modification. Defaults to None.
        n_particles (int, optional): Number of particles to estimate free energy with if using
        Gaussian variational distribution. Defaults to 1.

    Returns:
        List[float]: A dictionary with mean, min, and max batch loss over time.
    """
    mean_losses, min_losses, max_losses, layerwise_mean_losses = [], [], [], []
    prev_free_energy = None
    has_fixed_indices = fixed_indices_exists(fixed_indices=fixed_indices)
    if has_fixed_indices:
        a_group.clamp(obs=False, hidden=False)

    if activation_optim == 'adam':
        optim_cls = torch.optim.Adam
    elif activation_optim == 'sgd':
        optim_cls = torch.optim.SGD
    else:
        raise NotImplementedError()
    optimizer = optim_cls(a_group.data, lr=infer_lr)
    optimizer.zero_grad()

    for _ in range(infer_T):
        result = estimate_free_energy(log_joint_fn=log_joint_fn, a_group=a_group,
                                      n_particles=n_particles)
        free_energy = result.log_prob
        loss = free_energy.sum(dim=0)

        if loss.grad_fn is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if has_fixed_indices:
                orig_obs = a_group.original_obs
                pred_obs = a_group.get_acts(layer_index=0, detach=True)
                corrected_obs = orig_obs * fixed_indices + pred_obs * fixed_indices.logical_not()
                a_group.set_acts(layer_index=0, value=corrected_obs)

        # NOTE: min_losses/max_losses are misleading when batches are not i.i.d.
        #       so just set them to average loss as well if that's the case.
        mean_loss = loss.item() / a_group.d_batch
        min_loss = free_energy.min().item() if len(free_energy) > 1 else mean_loss
        max_loss = free_energy.max().item() if len(free_energy) > 1 else mean_loss
        layerwise_mean_loss = [l_layer / a_group.d_batch for l_layer in result.layer_log_probs]
        mean_losses.append(mean_loss)
        min_losses.append(min_loss)
        max_losses.append(max_loss)
        layerwise_mean_losses.append(layerwise_mean_loss)

        early_stop = early_stop_infer(free_energy=free_energy, prev_free_energy=prev_free_energy)
        prev_free_energy = free_energy.detach()
        if early_stop:
            break

    mean_losses.extend([mean_losses[-1]] * (infer_T - len(mean_losses)))
    min_losses.extend([min_losses[-1]] * (infer_T - len(min_losses)))
    max_losses.extend([max_losses[-1]] * (infer_T - len(max_losses)))
    a_group.clamp(obs=True, hidden=True)
    return {'mean_losses': mean_losses, 'min_losses': min_losses, 'max_losses': max_losses,
            'layerwise_mean_losses': layerwise_mean_losses}


def early_stop_infer(free_energy: torch.Tensor, prev_free_energy: torch.Tensor) -> bool:
    """Signal that inference iteration should stop if all differences between current and
    past log joint scores are less than 0.001.

    Args:
        free_energy (torch.Tensor): Current iteration free energy vector of shape <d_batch>.
        prev_free_energy (torch.Tensor): Past iteration free energy vector of shape <d_batch>.

    Returns:
        bool: Whether to stop inference iteration or not.
    """
    if prev_free_energy is None:
        return False
    return ((free_energy - prev_free_energy).abs() > 1e-3).sum() == 0


def ess_resample(objs: List[Any], log_weights: torch.Tensor, ess_thresh: float,
                 n_selected: int) -> Tuple[List[Any], torch.Tensor]:
    n_objects = len(objs)
    weights = log_weights.exp()
    ess = 1 / n_objects * (weights.sum().square() / weights.square().sum())
    if ess >= ess_thresh:
        return objs, log_weights
    log_weights = (torch.ones(n_objects) / n_objects).log().to(weights.device)
    indices = torch.multinomial(input=weights, num_samples=n_selected, replacement=True)
    objs = [deepcopy(objs[index]) for index in indices]
    return objs, log_weights


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
    x = torch.cat([torch.nn.functional.relu(X_in), torch.nn.functional.relu(-X_in)], dim=-1)
    # x = torch.cat([torch.nn.functional.elu(X_in)+1, torch.nn.functional.elu(-X_in)+1], dim=-1)
    x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, nu+1)], dim=-1)
    x_repeat = torch.cat([x] * nu, dim=-1)
    return x_repeat * x_rolled


def normalize(X_in: torch.Tensor) -> torch.Tensor:
    return X_in / torch.norm(X_in, p=2, dim=-1, keepdim=True)


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
