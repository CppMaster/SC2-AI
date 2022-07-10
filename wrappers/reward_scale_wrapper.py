import gym


class RewardScaleWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale
