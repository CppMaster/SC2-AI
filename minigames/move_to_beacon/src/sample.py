import random

from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop


class Agent(BaseAgent):
    pass


class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        unit = obs.observation.raw_units[0]
        return actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, (random.randint(0, 64), random.randint(0, 64)))


if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    agent = RandomAgent()
    try:
        with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=8,
                realtime=False,
                disable_fog=True,
        ) as env:
            run_loop.run_loop([agent], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass