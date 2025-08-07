from typing import Dict, Any, Union, Callable

import optuna
from torch import nn


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Create a linear learning rate schedule.
    
    This function returns a callable that implements a linear learning rate
    schedule where the learning rate decreases linearly with training progress.

    Parameters
    ----------
    initial_value : Union[float, str]
        Initial learning rate value. If string, will be converted to float.

    Returns
    -------
    Callable[[float], float]
        A function that takes progress_remaining and returns the learning rate.
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Calculate learning rate based on remaining progress.
        
        Progress will decrease from 1 (beginning) to 0 (end).

        Parameters
        ----------
        progress_remaining : float
            Remaining progress from 1.0 to 0.0.

        Returns
        -------
        float
            The learning rate for the current progress.
        """
        return progress_remaining * initial_value

    return func


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample PPO hyperparameters using Optuna trial.
    
    This function suggests hyperparameters for PPO training using Optuna's
    trial object. It covers a wide range of hyperparameters including
    learning rate, batch size, network architecture, and training parameters.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object for hyperparameter optimization.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the sampled hyperparameters for PPO.
    """
    # Core training parameters
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)

    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    
    # Network architecture parameters
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

    # Ensure batch_size doesn't exceed n_steps
    if batch_size > n_steps:
        batch_size = n_steps

    # Apply learning rate schedule if linear
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Network architecture configuration
    net_arch_width = trial.suggest_categorical("net_arch_width", [8, 16, 32, 64, 128, 256, 512])
    net_arch_depth = trial.suggest_int("net_arch_depth", 1, 3)
    net_arch = [dict(pi=[net_arch_width] * net_arch_depth, vf=[net_arch_width] * net_arch_depth)]

    # Convert activation function string to PyTorch module
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
