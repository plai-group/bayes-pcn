from abc import ABC, abstractmethod
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.distributions as dists
import pyro
import pyro.distributions as pdists

from bayes_pcn.const import *
from bayes_pcn.pcnet.util import is_PD, nearest_PD, gmm_log_marginal, gmm_log_posterior
from bayes_pcn.pcnet.kernels import get_kernel_sigmas, kernel_log_prob, kernel_posterior_params
from .activations import local_wta, dpfp


class AbstractPCLayer(ABC):

    def delete(self, X_obs: torch.Tensor, **kwargs) -> None:
        assert self._update_strat == LayerUpdateStrat.BAYES
        self._bayes_delete(X_obs=X_obs, **kwargs)

    @abstractmethod
    def log_prob(self, X_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def update(self, X_obs: torch.Tensor, **kwargs) -> None:
        if self._update_strat == LayerUpdateStrat.ML:
            self._ml_update(X_obs=X_obs, **kwargs)
        elif self._update_strat == LayerUpdateStrat.BAYES:
            self._bayes_update(X_obs=X_obs, **kwargs)
        elif self._update_strat == LayerUpdateStrat.KERNEL:
            self._kernel_update(X_obs=X_obs, **kwargs)
        elif self._update_strat == LayerUpdateStrat.NOISING:
            self._ml_update(X_obs=X_obs, **kwargs)
        else:
            raise NotImplementedError()

    @abstractmethod
    def param_norm(self, name: str) -> float:
        raise NotImplementedError()

    def _error(self, X_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return X_obs - self.predict(**kwargs)

    @abstractmethod
    def _ml_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _bayes_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _bayes_delete(self, X_obs: torch.Tensor, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def bayes_forget(self, beta_forget: float) -> None:
        raise NotImplementedError()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value
        for var in vars(self):
            if isinstance(self.__dict__[var], torch.Tensor):
                self.__dict__[var] = self.__dict__[var].to(value)

    @property
    def log_prob_strat(self):
        return self._log_prob_strat

    @log_prob_strat.setter
    def log_prob_strat(self, value: LayerLogProbStrat):
        self._log_prob_strat = value

    @property
    def sample_strat(self):
        return self._sample_strat

    @sample_strat.setter
    def sample_strat(self, value: LayerSampleStrat):
        self._sample_strat = value

    @property
    def update_strat(self):
        return self._update_strat

    @update_strat.setter
    def update_strat(self, value: LayerUpdateStrat):
        self._update_strat = value

    @abstractmethod
    def sample_parameters(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def fix_parameters(self, parameters: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def parameters_log_prob(self, parameters: torch.Tensor) -> float:
        raise NotImplementedError()


class PCLayer(AbstractPCLayer):
    """Maps upper layer output x_{l+1} to current layer output prediction x_{l}^
    Responsible for training weights responsible for this particular layer.
    """

    def __init__(self, layer_index: int, d_in: int, d_out: int, sigma_prior: float,
                 sigma_obs: float, act_fn: ActFn, **kwargs) -> None:
        self._name = f"layer_{layer_index}"
        self._d_in = d_in
        self._d_out = d_out
        self._log_prob_strat: LayerLogProbStrat = None
        self._sample_strat: LayerSampleStrat = None
        self._update_strat: LayerUpdateStrat = None
        self._device = torch.device('cpu')
        self._act_fn = act_fn
        self._weight_lr = kwargs.get('weight_lr', None)
        self._bias = kwargs.get('bias', None)

        if self._act_fn == ActFn.DPFP:
            self._d_in = 2 * self._d_in

        if self._bias:
            self._d_in = self._d_in + 1

        if kwargs.get('scale_layer', False):
            self._layer_norm = torch.nn.LayerNorm(self._d_in, elementwise_affine=False)
        else:
            self._layer_norm = None  # normalize

        if kwargs.get('weight_init_strat') == WeightInitStrat.FIXED:
            self._R = torch.zeros(self._d_in, self._d_out)
        else:
            self._R = torch.empty(self._d_in, self._d_out)
            torch.nn.init.kaiming_normal_(self._R, nonlinearity='linear')

        self._R_original = deepcopy(self._R)
        # MatrixNormal prior row-wise covariance matrix (initially isotropic)
        self._U = torch.eye(self._d_in) * sigma_prior ** 2
        self._U_original = deepcopy(self._U)
        # Isotropic observation variance
        self._Sigma = sigma_obs ** 2

        # Kernel related params
        self._X_train = None
        self._Y_train = None
        self._kernel_type = kwargs.get('kernel_type', Kernel.RBF)
        self._kernel_params = dict(Sigma_prior=sigma_prior**2, Sigma_obs=self._Sigma,
                                   lengthscale=1/sigma_prior)
        self._X_train_kernel = None

    def log_prob(self, X_obs: torch.Tensor, X_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return log probability under the model, log p(x_{l} | x_{l+1}).

        Args:
            X_obs (torch.Tensor): Observation matrix of shape <d_batch x d_out>.
            X_in (torch.Tensor): Input matrix of shape <d_batch x d_in>.

        Returns:
            torch.Tensor: Log probability vector of shape <d_batch>.
        """
        if self._update_strat == LayerUpdateStrat.KERNEL:
            # HACK: Create a new class for kernel stuff later
            return kernel_log_prob(X_train=self._X_train, Y_train=self._Y_train,
                                   X_test=X_in, Y_test=X_obs, kernel=self._kernel_type,
                                   kernel_params=self._kernel_params,
                                   Sigma_trtr_inv=self._X_train_kernel)

        Z_in = self._f(X_in)
        marginal_mean = self.predict(X_in=X_in)  # d_batch x d_out
        log_prob_strat = kwargs.get('log_prob_strat', self._log_prob_strat)
        if log_prob_strat == LayerLogProbStrat.MAP:
            marginal_Sigma = self._Sigma
        elif log_prob_strat == LayerLogProbStrat.P_PRED:
            d_batch = X_in.shape[0]
            if d_batch > 1 and not kwargs.get('batch_independence', False):
                # This is the proper thing to do when d_batch > 1: Batches are not independent.
                marginal_Sigma = self._Sigma * torch.eye(d_batch).to(self.device)\
                                + Z_in.matmul(self._U).matmul(Z_in.T)
                error_mean = torch.zeros(d_batch).to(self.device)
                dist = dists.MultivariateNormal(error_mean, marginal_Sigma)
                return dist.log_prob(X_obs.T - marginal_mean.T).sum().unsqueeze(0)
            marginal_Sigma = self._Sigma + Z_in.matmul(self._U).matmul(Z_in.T).diag().unsqueeze(-1)
        else:
            raise NotImplementedError()
        dist = dists.Normal(marginal_mean, marginal_Sigma ** 0.5)
        return dist.log_prob(X_obs).sum(dim=-1)

    def predict(self, X_in: torch.Tensor) -> torch.Tensor:
        return self._f(X_in).matmul(self._R)

    def sample(self, X_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample current layer neuron values given upper layer neuron values. When
        self._sample_strat is LayerSampleStrat.MAP and LayerSampleStrat.P_PRED, sample from
        p(X_out| argmax_W p(W), self._Sigma, ...) and E[p(X_out| W, self._Sigma, ...)].
        NOTE: Treats elements in batch independently.

        Args:
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.

        Returns:
            torch.Tensor: Sampled lower layer neuron values of shape <d_batch x d_out>.
        """
        if self._update_strat == LayerUpdateStrat.KERNEL:
            # HACK: Create a new class for kernel stuff later
            mus, Sigmas = kernel_posterior_params(X_train=self._X_train, Y_train=self._Y_train,
                                                  X_test=X_in, kernel=self._kernel_type,
                                                  kernel_params=self._kernel_params,
                                                  Sigma_trtr_inv=self._X_train_kernel)
            L = torch.linalg.cholesky(Sigmas)
            return mus + L.matmul(torch.randn_like(mus))

        marginal_mean = self.predict(X_in=X_in)
        if self._sample_strat == LayerSampleStrat.MAP:
            marginal_Sigma = self._Sigma
        elif self._sample_strat == LayerSampleStrat.P_PRED:
            Z_in = self._f(X_in)
            marginal_Sigma = self._Sigma + Z_in.matmul(self._U).matmul(Z_in.T).diag()
            marginal_Sigma = marginal_Sigma.unsqueeze(-1)
        else:
            raise NotImplementedError()
        dist = pdists.Normal(torch.zeros_like(marginal_mean), marginal_Sigma ** 0.5)
        obs = None if kwargs.get('X_obs') is None else (kwargs['X_obs']-marginal_mean)
        sample = marginal_mean + pyro.sample(self._name, dist, obs=obs)
        return sample

    def _f(self, X_in: torch.Tensor) -> torch.Tensor:
        # NOTE: To perform value normalization uncomment the line below and normalize
        # lower layer activations at pcnet.py update_weights function.
        # X_in = normalize(X_in=X_in)
        if self._act_fn == ActFn.NONE:
            result = X_in
        elif self._act_fn == ActFn.RELU:
            result = F.relu(X_in)
        elif self._act_fn == ActFn.GELU:
            result = F.gelu(X_in)
        elif self._act_fn == ActFn.SELU:
            result = F.selu(X_in)
        elif self._act_fn == ActFn.SOFTMAX:
            result = F.softmax(X_in, dim=-1)
        elif self._act_fn == ActFn.LWTA_SPARSE:
            # Neuron groups of size layer dim / 16 inhibit each other
            result = local_wta(X_in=X_in, block_size=X_in.shape[1]//16, hard=False)
        elif self._act_fn == ActFn.LWTA_DENSE:
            # Every neighbouring neurons inhibit each other
            result = local_wta(X_in=X_in, block_size=2, hard=False)
        elif self._act_fn == ActFn.DPFP:
            result = dpfp(X_in=X_in, nu=1)
        else:
            raise NotImplementedError()

        if self._layer_norm is not None:
            result = self._layer_norm(result)
        if self._bias:
            # Affine transform: intuitively equivalent to having a neuron that's always on
            result = torch.cat((result, torch.ones(X_in.shape[0], 1).to(X_in.device)), dim=-1)
        return result

    def _ml_update(self, X_obs: torch.Tensor, X_in: torch.Tensor, **kwargs) -> None:
        """Take a gradient step for the network parameters self._R. The gradient is
        averaged over the minibatch not summed.

        Args:
            X_obs (torch.Tensor): Lower layer neuron values of shape <d_batch x d_out>.
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.
        """
        d_batch = X_obs.shape[0]

        error = self._error(X_obs=X_obs, X_in=X_in)
        grad = self._f(X_in).T.matmul(error) / self._Sigma
        weight_lr = kwargs.get('lr', self._weight_lr)
        # print(grad.max())
        self._R = self._R + weight_lr / d_batch * grad

    def _bayes_update(self, X_obs: torch.Tensor, X_in: torch.Tensor) -> None:
        """Bayesian Multivariate linear regression update.

        Args:
            X_obs (torch.Tensor): Lower layer neuron values of shape <d_batch x d_out>.
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.
        """
        d_batch = X_obs.shape[0]
        orig_dtype = X_obs.dtype
        self._R, self._U = self._R.to(torch.float64), self._U.to(torch.float64)
        X_obs, X_in = X_obs.to(torch.float64), X_in.to(torch.float64)

        error = self._error(X_obs=X_obs, X_in=X_in)
        Z_in = self._f(X_in=X_in)
        Sigma_c = Z_in.matmul(self._U)
        Sigma_obs = self._Sigma * torch.eye(d_batch).to(self.device).to(torch.float64)
        Sigma_x = Sigma_c.matmul(Z_in.T) + Sigma_obs
        Sigma_x_inv = Sigma_x.inverse()
        Sigma_c_T_Sigma_x_inv = Sigma_c.T.matmul(Sigma_x_inv)

        self._R = self._R + Sigma_c_T_Sigma_x_inv.matmul(error)
        self._U = self._U - Sigma_c_T_Sigma_x_inv.matmul(Sigma_c)
        if not is_PD(self._U):
            self._U = nearest_PD(A=self._U.cpu()).to(Z_in.dtype).to(self._R.device)
        self._R, self._U = self._R.to(orig_dtype), self._U.to(orig_dtype)

    def _kernel_update(self, X_obs: torch.Tensor, X_in: torch.Tensor) -> None:
        """_summary_

        Args:
            X_obs (torch.Tensor): Lower layer neuron values of shape <d_batch x d_out>.
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.
        """
        self._X_train = X_in if self._X_train is None \
            else torch.cat([self._X_train, X_in], dim=0)
        self._Y_train = X_obs if self._Y_train is None \
            else torch.cat([self._Y_train, X_obs], dim=0)
        Sigma, _, _ = get_kernel_sigmas(X_train=self._X_train, X_test=None,
                                        kernel=self._kernel_type, kernel_params=self._kernel_params)
        self._X_train_kernel = Sigma.inverse()

    def _bayes_delete(self, X_obs: torch.Tensor, X_in: torch.Tensor) -> None:
        d_batch = X_obs.shape[0]
        orig_dtype = X_obs.dtype
        self._R, self._U = self._R.to(torch.float64), self._U.to(torch.float64)
        X_obs, X_in = X_obs.to(torch.float64), X_in.to(torch.float64)
        Z_in = self._f(X_in=X_in)

        self._U = (self._U.inverse() - 1/self._Sigma * Z_in.T.matmul(Z_in)).inverse()
        if not is_PD(self._U):
            self._U = nearest_PD(A=self._U.cpu()).to(Z_in.dtype).to(self._R.device)

        Sigma_c = Z_in.matmul(self._U)
        inv_term_R = Sigma_c.matmul(Z_in.T) + self._Sigma * torch.eye(d_batch).to(self.device)
        R_term_1 = torch.eye(self._d_in) + 1/self._Sigma * Sigma_c.T.matmul(Z_in)
        R_term_2 = self._R - Sigma_c.T.matmul(inv_term_R.inverse()).matmul(X_obs)
        self._R = R_term_1.matmul(R_term_2)
        self._R, self._U = self._R.to(orig_dtype), self._U.to(orig_dtype)

    def bayes_forget(self, beta_forget: float) -> None:
        self._R = (1-beta_forget)**0.5*self._R + (1-(1-beta_forget)**0.5)*self._R_original
        self._U = (1-beta_forget)*self._U + beta_forget*self._U_original
        # self._U = self._U + (beta_forget ** 2) * torch.eye(self._U.shape[0]).to(self.device)

    def sample_parameters(self) -> torch.Tensor:
        L = torch.linalg.cholesky(self._U)
        return self._R + L.matmul(torch.randn_like(self._R))

    def parameters_log_prob(self, parameters: torch.Tensor) -> float:
        n, p = self._R.shape
        error = parameters - self._R
        log_pdf = -0.5 * (error).matmul(self._U.inverse()).matmul(error).trace()
        log_normalizer = (n*p/2)*(2*torch.pi).log() + (p/2)*torch.logdet(self._U)
        return (log_pdf - log_normalizer).item()

    def fix_parameters(self, parameters: torch.Tensor):
        self._R = parameters

    def param_norm(self, name: str) -> float:
        if name == "R":
            param = self._R
        elif name == "U":
            param = self._U
        elif name == "U_diag":
            param = self._U.diag()
        else:
            raise NotImplementedError()
        return param.abs().mean().item()


class GaussianLayer(AbstractPCLayer):
    def __init__(self, layer_index: int, d_out: int, sigma_prior: float,
                 sigma_obs: float, **kwargs) -> None:
        self._name = f"layer_{layer_index}"
        self._d_out = d_out
        self._log_prob_strat: LayerLogProbStrat = None
        self._sample_strat: LayerSampleStrat = None
        self._update_strat: LayerUpdateStrat = None
        self._device = torch.device('cpu')
        self._weight_lr = kwargs.get('weight_lr', None)

        # Normal prior mean vector
        if kwargs.get('weight_init_strat') == WeightInitStrat.FIXED:
            self._R = torch.zeros(d_out)
        else:
            self._R = torch.empty(d_out)
            torch.nn.init.normal_(self._R, 0, d_out**-0.5)

        if kwargs.get('economy_mode', False):
            # HACK: Makes MHN memory efficient
            self._U = torch.eye(1) * sigma_prior ** 2
            self._Sigma = sigma_obs ** 2
            return

        self._R_original = deepcopy(self._R)
        # Normal prior covariance matrix
        self._U = torch.eye(d_out) * sigma_prior ** 2
        self._U_original = deepcopy(self._U)
        # Isotropic observation variance
        self._Sigma = sigma_obs ** 2

    def log_prob(self, X_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return log probability under the model, log p(x_{l} | x_{l+1}).

        Args:
            X_obs (torch.Tensor): Observation matrix of shape <d_batch x d_out>.

        Returns:
            torch.Tensor: Log probability vector of shape <d_batch>.
        """
        d_batch = X_obs.shape[0]
        marginal_mean = self.predict(d_batch=d_batch)
        log_prob_strat = kwargs.get('log_prob_strat', self._log_prob_strat)
        if log_prob_strat == LayerLogProbStrat.MAP:
            marginal_Sigma = self._Sigma
        elif log_prob_strat == LayerLogProbStrat.P_PRED:
            marginal_Sigma = self._Sigma + self._U[0, 0]
            if d_batch > 1 and not kwargs.get('batch_independence', True):
                dist = dists.Normal(marginal_mean, marginal_Sigma ** 0.5)
                return dist.log_prob(X_obs).sum().unsqueeze(-1)
        else:
            raise NotImplementedError()
        dist = dists.Normal(marginal_mean, marginal_Sigma ** 0.5)
        return dist.log_prob(X_obs).sum(dim=-1)

    def parameters_log_prob(self, parameters: torch.Tensor) -> float:
        dist = dists.MultivariateNormal(self._R, self._U)
        return dist.log_prob(parameters).item()

    def predict(self, X_in: torch.Tensor = None, d_batch: int = None, **kwargs) -> torch.Tensor:
        if d_batch is None:
            d_batch = 1 if X_in is None else X_in.shape[0]
        return self._R.repeat(d_batch, 1)

    def sample(self, **kwargs) -> torch.Tensor:
        """Sample current layer neuron values given upper layer neuron values. When
        self._sample_strat is LayerSampleStrat.MAP and LayerSampleStrat.P_PRED, sample from
        p(X_out| argmax_W p(W), self._Sigma, ...) and E[p(X_out| W, self._Sigma, ...)].

        Returns:
            torch.Tensor: Sampled lower layer neuron values of shape <d_batch x d_out>.
        """
        marginal_mean = self.predict(**kwargs)
        if self._sample_strat == LayerSampleStrat.MAP:
            marginal_Sigma = self._Sigma
        elif self._sample_strat == LayerSampleStrat.P_PRED:
            marginal_Sigma = self._Sigma + self._U[0, 0]
        else:
            raise NotImplementedError()
        dist = pdists.Normal(torch.zeros_like(marginal_mean), marginal_Sigma ** 0.5)
        obs = None if kwargs.get('X_obs') is None else (kwargs['X_obs']-marginal_mean)
        sample = marginal_mean + pyro.sample(self._name, dist, obs=obs)
        return sample

    def _ml_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        """Take a gradient step for the network parameters self._R. The gradient is
        averaged over the minibatch not summed.

        Args:
            X_obs (torch.Tensor): Observed neuron values of shape <d_batch x d_out>.
            lr (float, optional): Learning rate for layer weights.
        """
        d_batch = X_obs.shape[0]
        grad = self._error(X_obs=X_obs).sum(dim=0) / self._Sigma
        weight_lr = kwargs.get('lr', self._weight_lr)
        self._R = self._R + weight_lr / d_batch * grad

    def _kernel_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        self._bayes_update(X_obs=X_obs, **kwargs)

    def _bayes_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        """Bayesian normal normal conjugate update.

        Args:
            X_obs (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        d_batch = X_obs.shape[0]
        orig_dtype = X_obs.dtype
        self._R, self._U = self._R.to(torch.float64), self._U.to(torch.float64)
        X_obs = X_obs.to(torch.float64)
        mu_prior = self._R
        Sigma_prior_inv = 1 / self._U.diag()
        Sigma_obs_inv = 1 / self._Sigma * torch.ones(self._d_out).to(self.device).to(torch.float64)
        Sigma_posterior = (1 / (d_batch * Sigma_obs_inv + Sigma_prior_inv)).diag()
        mu_posterior = Sigma_posterior.matmul(Sigma_prior_inv * mu_prior +
                                              Sigma_obs_inv * X_obs.sum(dim=0))

        self._R = mu_posterior
        self._U = Sigma_posterior
        self._R, self._U = self._R.to(orig_dtype), self._U.to(orig_dtype)

    def _bayes_delete(self, X_obs: torch.Tensor, **kwargs) -> None:
        d_batch = X_obs.shape[0]
        orig_dtype = X_obs.dtype
        self._R, self._U = self._R.to(torch.float64), self._U.to(torch.float64)
        Sigma_posterior = self._U
        self._U = (1/self._U - d_batch/self._Sigma) ** -1
        self._R = self._U[0, 0]/Sigma_posterior[0, 0] * self._R\
            - self._U[0, 0]/self._Sigma * X_obs.mean(dim=0)
        self._R, self._U = self._R.to(orig_dtype), self._U.to(orig_dtype)

    def bayes_forget(self, beta_forget: float) -> None:
        self._R = (1-beta_forget)**0.5*self._R + (1-(1-beta_forget)**0.5)*self._R_original
        self._U = (1-beta_forget)*self._U + beta_forget*self._U_original
        # self._U = self._U + (beta_forget ** 2) * torch.eye(self._U.shape[0]).to(self.device)

    def sample_parameters(self) -> torch.Tensor:
        L = torch.linalg.cholesky(self._U)
        return self._R + L.matmul(torch.randn_like(self._R))

    def fix_parameters(self, parameters: torch.Tensor):
        self._R = parameters

    def param_norm(self, name: str) -> float:
        if name == "R":
            param = self._R
        elif name == "U":
            param = self._U
        elif name == "U_diag":
            param = self._U.diag()
        else:
            raise NotImplementedError()
        return param.abs().mean().item()


class KWayGMMLayer(AbstractPCLayer):
    def __init__(self, layer_index: int, K: int, n_components: int, d_out: int,
                 sigma_prior: float, sigma_obs: float, **kwargs) -> None:
        """
        Captures GMM layer at the top of the predictive coding network.

        Constructor Parameters
        - K: Number of parallel GMMs
        - n_components: Number of components in each GMM
        - d_out: Dimensionality of the continuous vector; should be divisible by K
        - Sigma_prior: Prior uncertainty over GMM component means
        - sigma_obs: GMM observation noise

        Should maintain K component importance vectors that can get reconfigured
        in the temporal setting.
        """
        assert d_out % K == 0 and K > 0
        self._name = f"layer_{layer_index}"
        self._K = K
        self._n_components = n_components
        self._d_out = d_out
        self._d_out_gmm = d_out // K

        self._log_prob_strat: LayerLogProbStrat = None
        self._sample_strat: LayerSampleStrat = None
        self._update_strat: LayerUpdateStrat = None
        self._device = torch.device('cpu')
        self._weight_lr = kwargs.get('weight_lr', None)
        self._sigma_prior = sigma_prior

        # Initialize K GMM parameters
        self._params = []
        for _ in range(K):
            # Component importance vector of shape <n_components>
            pi = torch.ones(n_components) / n_components
            # Matrix normal prior mean matrix of shape <n_components x (d_out / K)>
            if kwargs.get('weight_init_strat') == WeightInitStrat.FIXED:
                R = torch.zeros(n_components, self._d_out_gmm)
            else:
                R = torch.empty(n_components, self._d_out_gmm)
                torch.nn.init.normal_(R, 0, self._d_out_gmm**-0.5)
                # torch.nn.init.uniform_(R, -self._d_out_gmm**-0.5, self._d_out_gmm**-0.5)
            # Matrix normal prior covariance matrix of shape <n_components x n_components>
            U = torch.eye(n_components) * sigma_prior ** 2
            self._params.append(dict(pi=pi, R=R, U=U, z=None))

        # Isotropic observation variance
        self._Sigma = sigma_obs ** 2

        self._R_original = deepcopy(self._params[0]["R"])
        self._U_original = deepcopy(self._params[0]["U"])

    def log_prob(self, X_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return log probability under the model, log p(x^{l} | z^{1:K}).

        Args:
            X_obs (torch.Tensor): Observation matrix of shape <d_batch x d_out>.

        Returns:
            torch.Tensor: Log probability vector of shape <d_batch>.
        """
        result = 0.
        for k in range(self._K):
            params = self._params[k]
            X = X_obs[:, k*self._d_out_gmm:(k+1)*self._d_out_gmm]
            log_prob_strat = kwargs.get('log_prob_strat', self._log_prob_strat)
            # FIXME: P_PRED case does not support batch training yet
            U = params["U"] if log_prob_strat == LayerLogProbStrat.P_PRED else None
            curr_log_prob = gmm_log_marginal(X=X, pi=params["pi"], sigma_obs=self._Sigma**0.5,
                                             R=params["R"], U=U)
            result += curr_log_prob
        return result

    def predict(self, X_in: torch.Tensor = None, d_batch: int = None, **kwargs) -> torch.Tensor:
        """Returns the most likely GMM component means concatenated.

        Args:
            X_in (torch.Tensor, optional): _description_. Defaults to None.
            d_batch (int, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: Matrix of shape <d_batch x d_out>
        """
        if d_batch is None:
            d_batch = 1 if X_in is None else X_in.shape[0]
        result = torch.concat([param["R"][param["pi"].argmax().item()] for param in self._params])
        return result.repeat(d_batch, 1)

    def sample(self, **kwargs) -> torch.Tensor:
        """Sample current layer neuron values given upper layer neuron values. When
        self._sample_strat is LayerSampleStrat.MAP and LayerSampleStrat.P_PRED, sample from
        p(X_out| argmax_W p(W), self._Sigma, ...) and E[p(X_out| W, self._Sigma, ...)].

        Returns:
            torch.Tensor: Sampled lower layer neuron values of shape <d_batch x d_out>.
        """
        marginal_mean = self.predict(**kwargs)
        if self._sample_strat == LayerSampleStrat.MAP:
            marginal_Sigma = self._Sigma
        elif self._sample_strat == LayerSampleStrat.P_PRED:
            prior_vars = [p["U"][p["pi"].argmax().item()] for p in self._params]
            prior_vars = torch.concat([torch.ones(self._d_out_gmm)*pv for pv in prior_vars])
            marginal_Sigma = self._Sigma + prior_vars.to(marginal_mean.device)  # <d_out>
        else:
            raise NotImplementedError()
        dist = pdists.Normal(torch.zeros_like(marginal_mean), marginal_Sigma ** 0.5)
        obs = None if kwargs.get('X_obs') is None else (kwargs['X_obs']-marginal_mean)
        sample = marginal_mean + pyro.sample(self._name, dist, obs=obs)
        return sample

    def _mhn_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        # NOTE: Not called anywhere yet
        # Try adding components as observations are made, and increment self._n_components
        # Should support both individual and batch observation udpates
        d_batch, n_components = len(X_obs), self._n_components
        n_components_new = self._n_components + d_batch
        for k in range(self._K):
            params = self._params[k]
            pi, R, U = params["pi"], params["R"], params["U"]
            X = X_obs[:, k*self._d_out_gmm:(k+1)*self._d_out_gmm]
            log_posterior = gmm_log_posterior(X=X, pi=pi, sigma_obs=self._Sigma**0.5, R=R, U=U)
            params["R"] = torch.softmax(log_posterior, dim=0).T.matmul(X)

            old_mass, new_mass = self._n_components/n_components_new, d_batch/n_components_new
            old_pi, new_pi = pi*old_mass, torch.ones(d_batch).to(pi.device)/d_batch*new_mass
            pi_new = torch.cat((old_pi, new_pi), dim=0)
            R_new = torch.stack((R, X_obs), dim=0).to(R.device)
            if U is None:
                U_new = None
            else:
                U_new = torch.eye(n_components_new).to(U.device) * self._sigma_prior
                U_new[:n_components, :n_components] = U
            self._params[k] = dict(pi=pi_new, R=R_new, U=U_new, z=None)
        self._n_components += d_batch

    def _ml_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        """Expectation maximization of self._R given X_obs.
        Reference: https://en.wikipedia.org/wiki/EM_algorithm_and_GMM_model

        Args:
            X_obs (torch.Tensor): Observed neuron values of shape <d_batch x d_out>.
            lr (float, optional): Learning rate for layer weights.
        """
        for k in range(self._K):
            params = self._params[k]
            pi, R, U = params["pi"], params["R"], params["U"]
            X = X_obs[:, k*self._d_out_gmm:(k+1)*self._d_out_gmm]
            log_posterior = gmm_log_posterior(X=X, pi=pi, sigma_obs=self._Sigma**0.5, R=R, U=U)
            params["R"] = torch.softmax(log_posterior, dim=0).T.matmul(X)

    def _bayes_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        """Approximate Bayesian update of K GMMs. First, find the GMM component indices that
        most likely generated X_obs. Then, conditioned on X_obs and indices, perform conjugate
        Bayesian update on the network parameters.

        Args:
            X_obs (torch.Tensor): Data matrix of shape <d_batch x d_out>.
        """
        d_batch = X_obs.shape[0]
        for k in range(self._K):
            params = self._params[k]
            pi, R, U = params["pi"], params["R"], params["U"]
            X = X_obs[:, k*self._d_out_gmm:(k+1)*self._d_out_gmm]
            Z_in = torch.zeros(self._n_components).to(X_obs.device)
            Z_in[gmm_log_posterior(X=X, pi=pi, sigma_obs=self._Sigma**0.5, R=R, U=U).argmax()] = 1.
            print(gmm_log_posterior(X=X, pi=pi, sigma_obs=self._Sigma**0.5, R=R, U=U).argmax())

            Sigma_c = Z_in.unsqueeze(0).matmul(U)
            Sigma_obs = self._Sigma * torch.eye(d_batch).to(self.device)
            Sigma_x = Sigma_c.matmul(Z_in.unsqueeze(0).T) + Sigma_obs
            Sigma_x_inv = Sigma_x.inverse()
            Sigma_c_T_Sigma_x_inv = Sigma_c.T.matmul(Sigma_x_inv)

            R = R + Sigma_c_T_Sigma_x_inv.matmul(X - Z_in.matmul(R))
            U = U - Sigma_c_T_Sigma_x_inv.matmul(Sigma_c)
            self._params[k] = dict(pi=pi, R=R, U=U, z=Z_in)

    def bayes_forget(self, beta_forget: float) -> None:
        self._R = (1-beta_forget)**0.5*self._R + (1-(1-beta_forget)**0.5)*self._R_original
        self._U = (1-beta_forget)*self._U + beta_forget*self._U_original
        for k in range(self._K):
            params = self._params[k]
            R = (1-beta_forget)**0.5*params["R"] + (1-(1-beta_forget)**0.5)*self._R_original
            U = (1-beta_forget)*params["U"] + beta_forget*self._U_original
            self._params[k] = dict(pi=params["pi"], R=R, U=U, z=params["z"])

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value: torch.device):
        self._device = value
        for var in vars(self):
            if isinstance(self.__dict__[var], torch.Tensor):
                self.__dict__[var] = self.__dict__[var].to(value)
        for params in self._params:
            for k, v in params.items():
                if isinstance(v, torch.Tensor):
                    params[k] = v.to(value)

    def weight_norm(self, p: int = 1) -> float:
        return sum([params["R"].abs().mean().item() for params in self._params]) / self._K

    def param_norm(self, name: str) -> float:
        if name == "U_diag":
            return sum([params["U"].diag().abs().mean().item() for params in self._params])/self._K
        else:
            return sum([params[name].abs().mean().item() for params in self._params])/self._K

    def _bayes_delete(self, X_obs: torch.Tensor, **kwargs) -> None:
        raise NotImplementedError()

    def sample_parameters(self) -> torch.Tensor:
        raise NotImplementedError()

    def fix_parameters(self, parameters: torch.Tensor):
        raise NotImplementedError()

    def parameters_log_prob(self, parameters: torch.Tensor) -> float:
        raise NotImplementedError()
