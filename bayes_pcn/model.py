from typing import List, Tuple
import torch
import torch.nn.functional as F


class AbstractPCLayer:
    def predict(self, X_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(self, X_obs: torch.Tensor, X_in: torch.Tensor, lr: float = None) -> None:
        raise NotImplementedError

    def error(self, X_obs: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f(self, X_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f_grad(self, X_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: torch.device) -> None:
        raise NotImplementedError


class PCLayer(AbstractPCLayer):
    """Maps upper layer output x_{l+1} to current layer output prediction x_{l}^
    Responsible for training weights responsible for this particular layer. Uses
    ReLU activation by default.
    """

    def __init__(self, d_in: int, d_out: int, weight_lr: float) -> None:
        self.W = torch.empty(d_in, d_out)
        torch.nn.init.kaiming_normal_(self.W)
        self.weight_lr = weight_lr
        self.device = torch.device('cpu')

    def predict(self, X_in: torch.Tensor) -> torch.Tensor:
        """Predict lower layer activation based on upper layer activation.

        Args:
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.

        Returns:
            (torch.Tensor): Predicted lower layer neuron values of shape <d_batch x d_out>.
        """
        return self.f(X_in).matmul(self.W)

    def update(self, X_obs: torch.Tensor, X_in: torch.Tensor, lr: float = None) -> None:
        """Locally Hebbian weight update

        Args:
            X_obs (torch.Tensor): Lower layer neuron values of shape <d_batch x d_out>.
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.
            lr (float, optional): Learning rate for layer weights.
        """
        d_batch = X_obs.shape[0]
        if lr is None:
            lr = self.weight_lr

        error = self.error(X_obs, X_in)
        self.W += lr / d_batch * self.f(X_in).T.matmul(error)

    def error(self, X_obs: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
        return X_obs - self.predict(X_in)

    def f(self, X_in: torch.Tensor) -> torch.Tensor:
        return F.relu(X_in)

    def f_grad(self, X_in: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(X_in.shape).to(self.device)
        return ones * (X_in > torch.zeros(X_in.shape).to(self.device))

    def to(self, device: torch.device) -> None:
        self.device = device
        self.W = self.W.to(device)


class BayesPCLayer(PCLayer):
    """Maps upper layer output x_{l+1} to current layer output prediction x_{l}^
    Responsible for training weights responsible for this particular layer. Uses
    ReLU activation by default.
    """

    def __init__(self, d_in: int, d_out: int, weight_lr: float) -> None:
        # all covariance matrices are diagonal
        self.W = torch.empty(d_in, d_out)  # MatrixNormal prior mean matrix
        self.U = torch.eye(d_in) * 1e0    # MatrixNormal prior row-wise covariance matrix
        self.V = torch.eye(d_out) * 1e0   # MatrixNormal prior column-wise covariance matrix
        self.Sigma_obs = 1e0  # torch.eye(d_out) * 1e-1  # Observation covariance matrix

        torch.nn.init.kaiming_normal_(self.W)
        self.weight_lr = weight_lr
        self.device = torch.device('cpu')

    def update(self, X_obs: torch.Tensor, X_in: torch.Tensor, lr: float = None) -> None:
        """Locally Hebbian Bayesian weight update

        Args:
            X_obs (torch.Tensor): Lower layer neuron values of shape <d_batch x d_out>.
            X_in (torch.Tensor): Upper layer neuron values of shape <d_batch x d_in>.
            lr (float, optional): Learning rate for layer weights.
        """
        d_batch = X_obs.shape[0]

        error = self.error(X_obs, X_in)
        X_in = self.f(X_in=X_in)
        Sigma_c = X_in.matmul(self.U)
        Sigma_x = Sigma_c.matmul(X_in.T) + self.Sigma_obs * torch.eye(d_batch).to(self.device)
        Sigma_x_inv = Sigma_x.inverse()
        Sigma_c_T_Sigma_x_inv = Sigma_c.T.matmul(Sigma_x_inv)

        self.W = self.W + Sigma_c_T_Sigma_x_inv.matmul(error)
        self.U = self.U - Sigma_c_T_Sigma_x_inv.matmul(Sigma_c)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.W = self.W.to(device)
        self.U = self.U.to(device)
        self.V = self.V.to(device)
        # self.Sigma_obs = self.Sigma_obs.to(device)


class PCTopLayer(AbstractPCLayer):
    def __init__(self, d_mem: int, weight_lr: float) -> None:
        self.b = torch.randn(d_mem)
        self.d_mem = d_mem
        self.weight_lr = weight_lr
        self.device = torch.device('cpu')

    def predict(self, X_in: torch.Tensor = None) -> torch.Tensor:
        """Predict lower layer activation based on upper layer activation.

        Args:
            X_in (torch.Tensor, optional): If specified, return self.b repeated len(X_in)
                times horizontally for batch generation. Otherwise, assume repetition is
                not necessary.

        Returns:
            (torch.Tensor): Predicted lower layer neuron values of shape <d_batch x d_mem>.
        """
        d_batch = 1 if X_in is None else X_in.shape[0]
        return self.b.repeat(d_batch, 1)

    def update(self, X_obs: torch.Tensor, X_in: torch.Tensor, lr: float = None) -> None:
        """Locally Hebbian weight update

        Args:
            X_obs (torch.Tensor): Observed neuron values of shape <d_batch x d_mem>.
            X_in (torch.Tensor): Dummy value.
            lr (float, optional): Learning rate for layer weights.
        """
        if lr is None:
            lr = self.weight_lr
        self.b += lr * self.error(X_obs=X_obs).mean(dim=0)

    def error(self, X_obs: torch.Tensor, X_in: torch.Tensor = None) -> torch.Tensor:
        return X_obs - self.b

    def f(self, X_in: torch.Tensor) -> torch.Tensor:
        return X_in

    def f_grad(self, X_in: torch.Tensor) -> torch.Tensor:
        return torch.ones(X_in.shape).to(self.device)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.b = self.b.to(device)


class BayesPCTopLayer(PCTopLayer):
    def __init__(self, d_mem: int, weight_lr: float) -> None:
        # self.b = torch.randn(d_mem)
        self.b = torch.zeros(d_mem)
        self.Sigma_prior = torch.eye(d_mem) * 1e0  # Prior covariance matrix (diagonal)
        self.Sigma_obs = torch.eye(d_mem)  # Observation covariance matrix (diagonal)
        self.d_mem = d_mem
        self.weight_lr = weight_lr
        self.device = torch.device('cpu')

    def update(self, X_obs: torch.Tensor, X_in: torch.Tensor, lr: float = None) -> None:
        """Locally Hebbian Bayesian weight update
        Refer to https://stats.stackexchange.com/questions/28744/multivariate-normal-posterior
        NOTE: It would be interesting to enable individually weighted regression here

        Args:
            X_obs (torch.Tensor): Observed neuron values of shape <d_batch x d_mem>.
            X_in (torch.Tensor): Dummy value.
            lr (float, optional): Learning rate for layer weights.
        """
        d_batch = X_obs.shape[0]
        mu_prior = self.b
        Sigma_prior_inv, Sigma_obs_inv = 1/self.Sigma_prior.diag(), 1/self.Sigma_obs.diag()
        Sigma_posterior = (1 / (d_batch * Sigma_obs_inv + Sigma_prior_inv)).diag()
        mu_posterior = Sigma_posterior.matmul(
                            mu_prior * Sigma_prior_inv + Sigma_obs_inv * X_obs.sum(dim=0))

        self.b = mu_posterior
        self.Sigma_prior = Sigma_posterior

    def to(self, device: torch.device) -> None:
        self.device = device
        self.b = self.b.to(device)
        self.Sigma_prior = self.Sigma_prior.to(device)
        self.Sigma_obs = self.Sigma_obs.to(device)


class PCNet:
    """Predictive Coding Network class. Learns via Inference Learning. Can denoise
    noisy data or generate data. Composed of multiple neuron layers, with 0th layer
    neuron activations representing the data.

    NOTE: self.layers is a list of objects that encapsulate synaptic weights and
    nonlinearities that allows upper layer neuron activations to predict lower layer
    neuron activations. It is ordered s.t. the first element maps 1st layer activations
    to 0th layer activations, the second element maps 2nd layer activations to 1st
    layer activations, and so on.
    """

    def __init__(self, mode: str, n_layers: int, d_out: int, d_h: int, weight_lr: float,
                 activation_lr: float, T_infer: int, n_repeat: int) -> None:
        """_summary_

        Args:
            mode (str): One of 'ml' or 'bayes'. Determines how weights are updated.
            n_layers (int): Number of neuron layers in the model. Includes the 0th
            neuron layer that represents data.
            d_out (int): Dimension of the 0th neuron layer.
            d_h (int): Dimension of all other neuron layers aside from the 0th layer.
            weight_lr (float): Learning rate for synaptic weights.
            activation_lr (float): Learning rate for neuron activations.
            T_infer (int): Number of activation gradient descent iterations.
            n_repeat (int): Number of times self._infer is repeated before learning.
        """
        assert n_layers > 1
        self.n_layers = n_layers
        self.d_h = d_h
        self.d_out = d_out
        self.weight_lr = weight_lr
        self.activation_lr = activation_lr
        self.T_infer = T_infer
        self.n_repeat = n_repeat
        self.device = torch.device('cpu')

        # Populate bottom, intermediate, and top layer mappings
        if mode == 'ml':
            self.layers = [PCLayer(d_in=d_h, d_out=d_out, weight_lr=weight_lr)]
            for _ in range(self.n_layers-2):
                self.layers.append(PCLayer(d_in=d_h, d_out=d_h, weight_lr=weight_lr))
            self.layers.append(PCTopLayer(d_mem=d_h, weight_lr=weight_lr))
        elif mode == 'bayes':
            self.layers = [BayesPCLayer(d_in=d_h, d_out=d_out, weight_lr=weight_lr)]
            for _ in range(self.n_layers-2):
                self.layers.append(BayesPCLayer(d_in=d_h, d_out=d_h, weight_lr=weight_lr))
            self.layers.append(BayesPCTopLayer(d_mem=d_h, weight_lr=weight_lr))
        else:
            raise Exception(f"mode should be either 'ml' or 'bayes'.")

    def generate_ancestral(self, d_batch: int = 1, noise: float = 1.,
                           ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generates data similar to previously observed datapoints through
        ancestral sampling.

        Args:
            d_batch (int, optional): Number of data to generate. Defaults to 1.

        Returns:
            torch.Tensor: Predicted mean of the output distribution.
        """
        activations = []
        X = torch.empty(d_batch, self.d_h).to(self.device)
        for layer in reversed(self.layers[1:]):
            X = layer.predict(X_in=X) + torch.randn(X.shape).to(self.device) * noise
            activations.append(X)
        X_pred = self.layers[0].predict(X_in=X)
        activations = [X_pred + torch.randn(X_pred.shape).to(self.device) * noise] + activations
        return X_pred, activations

    def generate_iterative(self, d_batch: int = 1, noise: float = 1., n_repeat: int = None,
                           ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generates data similar to previously observed datapoints through
        ancestral sampling.

        Args:
            d_batch (int, optional): Number of data to generate. Defaults to 1.

        Returns:
            torch.Tensor: Predicted mean of the output distribution.
        """
        X, activations = self.generate_ancestral(d_batch=d_batch, noise=noise)
        if n_repeat is None:
            n_repeat = self.n_repeat
        for _ in range(n_repeat):
            X, activations = self._infer(X_obs=X, activations=activations)
            activations[0] = X
        return X, activations

    def learn(self, X_obs: torch.Tensor) -> float:
        activations = None
        for _ in range(self.n_repeat):
            _, activations = self._infer(X_obs=X_obs.to(self.device), activations=activations)
        # _, activations = self._infer(X_obs=X_obs.to(self.device))
        self._update(activations=activations)

    def infer(self, X_obs: torch.Tensor, n_repeat: int = None,
              fixed_indices: torch.Tensor = None) -> torch.Tensor:
        original_device = X_obs.device
        X_obs, activations = X_obs.to(self.device), None
        if n_repeat is None:
            n_repeat = self.n_repeat
        if fixed_indices is not None:
            fixed_indices = fixed_indices.to(self.device)

        for _ in range(n_repeat):
            X_pred, activations = self._infer(X_obs=X_obs, activations=activations,
                                              fixed_indices=fixed_indices)
            activations[0] = X_pred
        return X_pred.to(original_device)

    def to(self, device: torch.device) -> None:
        """Perform tensor computations on GPU.

        Args:
            device (torch.device): PyTorch device object.
        """
        self.device = device
        for layer in self.layers:
            layer.to(device)

    def _infer(self, X_obs: torch.Tensor, activations: List[torch.Tensor] = None,
               fixed_indices: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Given some input which is either a novel datapoint or a noisy version of
        previously seen data, return the model's best denoised guess of the input.
        First, fix 0th layer neuron activations. Then, repeatedly adjust hidden layer
        activations to minimize the global energy. Lastly, return the model's prediction
        of 0th neuron layer activations.

        Args:
            X_obs (torch.Tensor): Input data of shape <d_batch x d_out>.
            activations (List[torch.Tensor], optional): List of layer activations to start from.
            fixed_indices (torch.Tensor): Boolean matrix of shape <d_batch x d_out> that denotes
                which X_obs indices to prevent modification.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Returns the denoised input data and
                a list of network activations that most likely generated the input data.
        """
        lr = self.activation_lr
        d_batch = X_obs.shape[0]
        if activations is None:
            activations = [X_obs] + self._init_hidden_activations(d_batch=d_batch)
        has_fixed_indices = fixed_indices is not None and fixed_indices.max() == 1

        # update hidden layer activations
        prev_batch_joint_mse = None
        for t in range(1, self.T_infer+1):
            batch_joint_mse = torch.zeros(d_batch).to(self.device)
            for i in range(1, len(activations)):
                # upper_activation is not needed for top level activation update.
                upper_activation = activations[i+1] if i+1 < len(activations) else None
                activation = activations[i]
                lower_activation = activations[i-1]
                upper_layer = self.layers[i]
                lower_layer = self.layers[i-1]

                upper_error = upper_layer.error(X_obs=activation, X_in=upper_activation)
                lower_error = lower_layer.error(X_obs=lower_activation, X_in=activation)

                lower_f_grad = lower_layer.f_grad(activation)
                joint_grad = lower_f_grad * lower_error.matmul(lower_layer.W.T) - upper_error
                activations[i] = activations[i] + lr * joint_grad

                if i == 1:
                    batch_joint_mse += 0.5 * lower_error.norm(p=2, dim=-1) ** 2
                batch_joint_mse += 0.5 * upper_error.norm(p=2, dim=-1) ** 2

            if has_fixed_indices:
                X_pred = self.layers[0].predict(activations[1])
                activations[0] = X_pred * fixed_indices.logical_not() + X_obs * fixed_indices

            # if t == 1 or t == self.T_infer:
            #     print(f"Batch Joint MSE ({t}): {batch_joint_mse}")
            if t > 1 and (torch.abs(batch_joint_mse - prev_batch_joint_mse) > 1e-3).sum() == 0:
                break
            else:
                prev_batch_joint_mse = batch_joint_mse

        X_pred = self.layers[0].predict(activations[1])
        if has_fixed_indices:
            # fix indices in X_pred to values in X_obs if necessary
            X_pred = X_pred * fixed_indices.logical_not() + X_obs * fixed_indices
        return X_pred, activations

    def _update(self, activations: List[torch.Tensor]) -> None:
        """Performs a single gradient update of PCNet weights. Equivalent to minimizing the
        global energy function w.r.t. PCN weights. Must be called with the output of self.infer().

        Args:
            activations (List[torch.Tensor]): Output of self.infer(), which is the list of
                network activations activations that most likely generated the input data
                (the first element of the list).
        """
        lr = self.weight_lr
        for i, layer in enumerate(self.layers):
            # upper_activation is not needed for top level weight update.
            upper_activation = activations[i+1] if i+1 < len(activations) else None
            lower_activation = activations[i]
            layer.update(X_obs=lower_activation, X_in=upper_activation, lr=lr)

    def _init_hidden_activations(self, d_batch: int = 1) -> torch.Tensor:
        # return self.generate_ancestral(d_batch=d_batch)[1][1:]
        # return [torch.randn(d_batch, self.d_h).to(self.device) for _ in range(self.n_layers-1)]
        # return [torch.zeros(d_batch, self.d_h).to(self.device) for _ in range(self.n_layers-1)]
        return [0.1 * torch.randn(d_batch, self.layers[i].W.shape[0]).to(self.device)
                for i in range(self.n_layers-1)]
