from typing import Set, Optional, Callable, List, Union, Dict, Type, Any

import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution, \
    MultiCategoricalDistribution, BernoulliDistribution, StateDependentNoiseDistribution, Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class ActorCriticActionEliminationPolicy(ActorCriticPolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            eliminated_actions_func: Optional[Callable[[], Set[int]]] = None
    ):
        super().__init__(
            observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init,
            full_std, sde_net_arch, use_expln, squash_output, features_extractor_class, features_extractor_kwargs,
            normalize_images, optimizer_class, optimizer_kwargs
        )
        self.eliminated_actions_func: Optional[Callable[[], Set[int]]] = eliminated_actions_func

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if self.eliminated_actions_func:
            mask_tensor = th.tensor(
                [-1e6 if i in self.eliminated_actions_func() else 0.0 for i in range(mean_actions.shape[1])],
                device=mean_actions.device
            )
            mean_actions = th.add(mean_actions, mask_tensor)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")
