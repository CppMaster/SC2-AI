import logging

from minigames.simple_map.src.planned_action_env.reward_shaper import RewardShaper


class SupplyDepotRewardShaper(RewardShaper):
    """
    Reward shaper for supply depot construction and supply management.

    Attributes
    ----------
    reward_diff : float
        Reward difference scaling factor.
    free_supply_margin_factor : float
        Margin factor for free supply.
    last_supply_depot_index : int
        Last supply depot index recorded.
    logger : logging.Logger
        Logger for this shaper.
    """
    def __init__(self, reward_diff: float = 0.1, free_supply_margin_factor: float = 1.5) -> None:
        """
        Initialize the SupplyDepotRewardShaper.

        Parameters
        ----------
        reward_diff : float, optional
            Reward difference scaling factor (default is 0.1).
        free_supply_margin_factor : float, optional
            Margin factor for free supply (default is 1.5).
        """
        super().__init__()
        self.logger = logging.getLogger("SupplyDepotRewardShaper")
        self.reward_diff = reward_diff
        self.free_supply_margin_factor = free_supply_margin_factor
        self.last_supply_depot_index = 0

    def reset(self) -> None:
        """
        Reset the reward shaper state at the beginning of an episode.
        """
        self.last_supply_depot_index = 0

    def get_shaped_reward(self) -> float:
        """
        Compute the shaped reward for supply depot construction.

        Returns
        -------
        float
            The shaped reward value.
        """
        shaped_reward = 0.0
        supply_depot_index = self.env.supply_depot_index
        if supply_depot_index > self.last_supply_depot_index:
            expected_supply_cap = self.env.get_expected_supply_cap() - 8
            supply_taken = self.env.get_supply_taken() + self.free_supply_margin_factor * (
                1 + self.env.get_production_building_count()
            )
            shaped_reward = (supply_taken - expected_supply_cap) * self.reward_diff
            self.logger.debug(f"Supply taken: {supply_taken}, Expected supply cap: {expected_supply_cap}, "
                              f"Shaped reward: {shaped_reward}")
        self.last_supply_depot_index = supply_depot_index
        return shaped_reward
