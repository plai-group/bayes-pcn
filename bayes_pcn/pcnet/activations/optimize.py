import torch
from typing import Callable, Dict, List
from . import ActivationGroup
from bayes_pcn.pcnet.structs import LogProbResult
from bayes_pcn.pcnet.util import fixed_indices_exists


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
    layer_log_joints = [-lp for lp in lp_result.layer_log_probs]
    # print(layer_log_joints[0])
    return LogProbResult(log_prob=-elbo, layer_log_probs=layer_log_joints)


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
