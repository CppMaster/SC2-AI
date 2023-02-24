from typing import Tuple

import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.optim import Adam


class ImageToImagePolicy(nn.Module):

    def __init__(self, n_conv_layers: int = 5, kernel_size: int = 3, value_features_layer: int = 3,
                 n_input_channels: int = 3, n_hidden_channels: int = 16, n_output_channels: int = 2,
                 n_value_layers: int = 2, n_value_neurons: int = 64, grid_width: int = 16, grid_height: int = 16):
        super().__init__()
        self.action_dim = grid_width * grid_height
        self.n_output_channels = n_output_channels
        self.conv_layers = nn.ModuleList([nn.Conv2d(
            in_channels=n_hidden_channels if i > 0 else n_input_channels,
            out_channels=n_hidden_channels if i < n_conv_layers - 1 else n_output_channels,
            kernel_size=kernel_size,
            padding="same"
        ) for i in range(n_conv_layers)])
        self.value_layers = nn.ModuleList([nn.Linear(
            in_features=n_value_neurons if i > 0 else self.action_dim * n_hidden_channels,
            out_features=n_value_neurons if i < n_value_layers - 1 else 1
        ) for i in range(n_value_layers)])
        self.value_features_layer = value_features_layer
        self.flatten_action = nn.Flatten(start_dim=2, end_dim=3)
        self.action_distribution = CategoricalDistribution(self.action_dim)
        self.optimizer = Adam(self.parameters())
        self.device = torch.device("cuda")

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        features = self.extract_features(obs)
        latent_pi = self.get_latent_pi(features)
        latent_pi_flattened = self.flatten_action(latent_pi)
        distributions = [self.action_distribution.proba_distribution(latent_pi_flattened[:, i])
                         for i in range(self.n_output_channels)]
        actions = [distribution.get_actions(deterministic=deterministic) for distribution in distributions]
        log_probs = [distribution.log_prob(action) for action, distribution in zip(actions, distributions)]
        actions = torch.stack(actions, -1)
        log_probs = torch.stack(log_probs, -1)

        values = self.get_values(features)

        return actions, values, log_probs

    def reset_noise(self, n_envs: int = 1) -> None:
        pass

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        features = obs
        for i in range(self.value_features_layer):
            features = self.conv_layers[i](features)
            features = F.leaky_relu(features)
        return features

    def get_values(self, features: torch.Tensor) -> torch.Tensor:
        values = torch.flatten(features, start_dim=1)
        for i in range(len(self.value_layers)):
            values = self.value_layers[i](values)
            if i < len(self.value_layers) - 2:
                values = F.leaky_relu(values)
        return values

    def get_latent_pi(self, features: torch.Tensor) -> torch.Tensor:
        latent_pi = features
        for i in range(self.value_features_layer, len(self.conv_layers)):
            latent_pi = self.conv_layers[i](latent_pi)
        return latent_pi

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi = self.get_latent_pi(features)
        latent_pi_flattened = self.flatten_action(latent_pi)
        distributions = [self.action_distribution.proba_distribution(latent_pi_flattened[:, i])
                         for i in range(self.n_output_channels)]
        log_probs = [distributions[i].log_prob(actions[:, i]) for i in range(self.n_output_channels)]
        entropies = [distribution.entropy() for distribution in distributions]
        log_probs = torch.stack(log_probs, -1)
        entropies = torch.stack(entropies, -1)

        values = self.get_values(features)

        return values, log_probs, entropies

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def obs_to_tensor(self, observation: np.ndarray) -> Tuple[torch.Tensor, bool]:
        observation = obs_as_tensor(observation, self.device)
        return observation, False

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.get_values(self.extract_features(obs))
