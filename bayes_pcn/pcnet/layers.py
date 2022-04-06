from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import torch.distributions as dists

from bayes_pcn.const import *


class AbstractPCLayer(ABC):

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
        else:
            raise NotImplementedError()

    def _error(self, X_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return X_obs - self.predict(**kwargs)

    @abstractmethod
    def _ml_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _bayes_update(self, X_obs: torch.Tensor, **kwargs) -> None:
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


class PCLayer(AbstractPCLayer):
    """Maps upper layer output x_{l+1} to current layer output prediction x_{l}^
    Responsible for training weights responsible for this particular layer.
    """

    def __init__(self, d_in: int, d_out: int, sigma_prior: float, sigma_obs: float,
                 act_fn: ActFn, **kwargs) -> None:
        self._d_in = d_in
        self._d_out = d_out
        self._log_prob_strat: LayerLogProbStrat = None
        self._sample_strat: LayerSampleStrat = None
        self._update_strat: LayerUpdateStrat = None
        self._device = torch.device('cpu')
        self._act_fn = act_fn
        self._weight_lr = kwargs.get('weight_lr', None)
        self._scale = kwargs.get('scale_layer', False)

        # MatrixNormal prior mean matrix
        self._R = torch.empty(d_in, d_out)
        torch.nn.init.kaiming_uniform_(self._R, a=5**0.5)  # default torch initialization
        # MatrixNormal prior row-wise covariance matrix (initially isotropic)
        self._U = torch.eye(d_in) * sigma_prior ** 2
        # Isotropic observation variance
        self._Sigma = sigma_obs ** 2

    def log_prob(self, X_obs: torch.Tensor, X_in: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return log probability under the model, log p(x_{l} | x_{l+1}).

        Args:
            X_obs (torch.Tensor): Observation matrix of shape <d_batch x d_out>.
            X_in (torch.Tensor): Input matrix of shape <d_batch x d_in>.

        Returns:
            torch.Tensor: Log probability vector of shape <d_batch>.
        """
        # TODO: Consider normalizing Z_in by 1/sqrt(dim) like transformers
        # so dimensionality matters less in Z_in U Z_in' (Z_in Z_in' has norm 1)
        # if they are both normally distributed
        Z_in = self._f(X_in)
        marginal_mean = self.predict(X_in=X_in)  # d_batch x d_out
        log_prob_strat = kwargs.get('log_prob_strat', self._log_prob_strat)
        if log_prob_strat == LayerLogProbStrat.MAP:
            marginal_Sigma = self._Sigma
        elif log_prob_strat == LayerLogProbStrat.P_PRED:
            marginal_Sigma = (self._Sigma + Z_in.matmul(self._U).matmul(Z_in.T).diag()).unsqueeze(1)
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
        d_batch = X_in.shape[0]
        marginal_mean = self.predict(X_in=X_in)
        white_noise = torch.randn(d_batch, self._d_out).to(self.device)

        if self._sample_strat == LayerSampleStrat.MAP:
            noise = (self._Sigma ** 0.5) * white_noise
        elif self._sample_strat == LayerSampleStrat.P_PRED:
            Z_in = self._f(X_in)
            marginal_Sigma = self._Sigma + Z_in.matmul(self._U).matmul(Z_in.T).diag()
            noise = (marginal_Sigma ** 0.5).unsqueeze(-1) * white_noise
        else:
            raise NotImplementedError()
        return marginal_mean + noise

    def _f(self, X_in: torch.Tensor) -> torch.Tensor:
        # NOTE: Rescale input by 1/sqrt(d)
        if self._act_fn == ActFn.NONE:
            result = X_in
        elif self._act_fn == ActFn.RELU:
            result = F.relu(X_in)
        elif self._act_fn == ActFn.GELU:
            result = F.gelu(X_in)
        elif self._act_fn == ActFn.SOFTMAX:
            result = F.softmax(X_in, dim=-1)
        else:
            raise NotImplementedError()
        scaling = 1/(X_in.shape[-1] ** 0.5) if self._scale else 1
        return result * scaling

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
        self._R = self._R + weight_lr / d_batch * grad

    def _bayes_update(self, X_obs: torch.Tensor, X_in: torch.Tensor) -> None:
        """Bayesian Multivariate linear regression update.

        Args:
            X_obs (torch.Tensor): Lower layer neuron values of shape <d_batch x d_out>.
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.
            lr (float, optional): Learning rate for layer weights.
        """
        d_batch = X_obs.shape[0]

        error = self._error(X_obs=X_obs, X_in=X_in)
        Z_in = self._f(X_in=X_in)
        Sigma_c = Z_in.matmul(self._U)
        Sigma_x = Sigma_c.matmul(Z_in.T) + self._Sigma * torch.eye(d_batch).to(self.device)
        Sigma_x_inv = Sigma_x.inverse()
        Sigma_c_T_Sigma_x_inv = Sigma_c.T.matmul(Sigma_x_inv)

        self._R = self._R + Sigma_c_T_Sigma_x_inv.matmul(error)
        self._U = self._U - Sigma_c_T_Sigma_x_inv.matmul(Sigma_c)


class PCTopLayer(AbstractPCLayer):
    def __init__(self, d_out: int, sigma_prior: float, sigma_obs: float, **kwargs) -> None:
        self._d_out = d_out
        self._log_prob_strat: LayerLogProbStrat = None
        self._sample_strat: LayerSampleStrat = None
        self._update_strat: LayerUpdateStrat = None
        self._device = torch.device('cpu')
        self._weight_lr = kwargs.get('weight_lr', None)

        # Normal prior mean vector
        self._R = torch.empty(d_out)
        torch.nn.init.uniform_(self._R, -d_out**-0.5, d_out**-0.5)
        # Normal prior covariance matrix
        self._U = torch.eye(d_out) * sigma_prior ** 2
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
        else:
            raise NotImplementedError()
        dist = dists.Normal(marginal_mean, marginal_Sigma ** 0.5)
        return dist.log_prob(X_obs).sum(dim=-1)

    def predict(self, X_in: torch.Tensor = None, d_batch: int = None) -> torch.Tensor:
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
        mean = self.predict(**kwargs)
        d_batch = mean.shape[0]
        white_noise = torch.randn(d_batch, self._d_out).to(self.device)

        if self._sample_strat == LayerSampleStrat.MAP:
            noise = (self._Sigma ** 0.5) * white_noise
        elif self._sample_strat == LayerSampleStrat.P_PRED:
            noise = ((self._Sigma + self._U[0, 0]) ** 0.5) * white_noise
        else:
            return NotImplementedError()
        return mean + noise

    def _ml_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        """Take a gradient step for the network parameters self._R. The gradient is
        averaged over the minibatch not summed.

        Args:
            X_obs (torch.Tensor): Observed neuron values of shape <d_batch x d_out>.
            lr (float, optional): Learning rate for layer weights.
        """
        weight_lr = kwargs.get('lr', self._weight_lr)
        self._R = self._R + weight_lr * self._error(X_obs=X_obs).mean(dim=0)

    def _bayes_update(self, X_obs: torch.Tensor, **kwargs) -> None:
        """Bayesian normal normal conjugate update.

        Args:
            X_obs (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        d_batch = X_obs.shape[0]
        mu_prior = self._R
        Sigma_prior_inv = 1 / self._U.diag()
        Sigma_obs_inv = 1 / self._Sigma * torch.ones(self._d_out).to(self.device)
        Sigma_posterior = (1 / (d_batch * Sigma_obs_inv + Sigma_prior_inv)).diag()
        mu_posterior = Sigma_posterior.matmul(Sigma_prior_inv * mu_prior +
                                              Sigma_obs_inv * X_obs.sum(dim=0))

        self._R = mu_posterior
        self._U = Sigma_posterior