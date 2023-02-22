import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import nn


class ImageToImagePolicy(nn.Module):

    def __init__(self, n_conv_layers: int = 5, kernel_size: int = 3, value_features_layer: int = 3,
                 n_input_channels: int = 3, n_hidden_channels: int = 16, n_output_channels: int = 2,
                 n_value_layers: int = 2, n_value_neurons: int = 64, grid_width: int = 16, grid_height: int = 16):
        super().__init__()
        self.action_dim = grid_width * grid_height
        self.n_output_channels = n_output_channels
        self.conv_layers = [nn.Conv2d(
            in_channels=n_hidden_channels if i > 0 else n_input_channels,
            out_channels=n_hidden_channels if i < n_conv_layers - 1 else n_output_channels,
            kernel_size=kernel_size,
            padding="same"
        ) for i in range(n_conv_layers)]
        self.value_layers = [nn.Linear(
            in_features=n_value_neurons if i > 0 else self.action_dim * n_hidden_channels,
            out_features=n_value_neurons if i < n_value_layers - 1 else 1
        ) for i in range(n_value_layers)]
        self.value_features_layer = value_features_layer
        self.flatten_action = nn.Flatten(start_dim=1, end_dim=2)
        self.action_distribution = CategoricalDistribution(self.action_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        features = obs
        for i in range(self.value_features_layer):
            features = self.conv_layers[i](features)

        latent_pi = features
        for i in range(self.value_features_layer, len(self.conv_layers)):
            latent_pi = self.conv_layers[i](latent_pi)
        latent_pi_flattened = self.flatten_action(latent_pi)
        distributions = [self.action_distribution.proba_distribution(latent_pi_flattened[:, :, i])
                         for i in range(self.n_output_channels)]
        actions = [distribution.get_actions(deterministic=deterministic) for distribution in distributions]
        log_probs = [distribution.log_prob(action) for action, distribution in zip(actions, distributions)]
        actions = torch.stack(actions, -1)
        log_probs = torch.stack(log_probs, -1)

        values = torch.flatten(features, start_dim=1)
        for i in range(len(self.value_layers)):
            values = self.value_layers[i](values)

        return actions, values, log_probs


