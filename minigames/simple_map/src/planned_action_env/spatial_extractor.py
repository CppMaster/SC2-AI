import torch as th
from torch import nn
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialExtractor(BaseFeaturesExtractor):
    """
    Spatial feature extractor using CNN for processing minimap observations.

    This extractor processes spatial data (like minimap observations) through
    a series of convolutional layers to extract meaningful spatial features.

    Parameters
    ----------
    observation_space : spaces.Box
        The observation space containing spatial data.
    features_dim : int, optional
        Number of features extracted (default is 256).
        This corresponds to the number of units for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256) -> None:
        """
        Initialize the SpatialExtractor.

        Parameters
        ----------
        observation_space : spaces.Box
            The observation space containing spatial data.
        features_dim : int, optional
            Number of features extracted (default is 256).
        """
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the spatial extractor.

        Parameters
        ----------
        observations : th.Tensor
            Input observations tensor.

        Returns
        -------
        th.Tensor
            Extracted spatial features.
        """
        return self.linear(self.cnn(observations))
