import gym
import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from minigames.simple_map.src.planned_action_env.spatial_extractor import SpatialExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, spatial_feature_dim=256):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "minimap":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = SpatialExtractor(subspace, features_dim=spatial_feature_dim)
                total_concat_size += spatial_feature_dim
            elif key == "non_spatial":
                # Run through a simple MLP
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)