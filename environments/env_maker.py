import gym

from environments.locomotion import JerkAnt, JerkHumanoid


def make_env(config):

    if config.env_name == 'JerkAnt':
        env = JerkAnt(penalize_jerk=config.penalize_jerk, jerk_weight=config.jerk_weight)
    elif config.env_name == 'JerkHumanoid':
        env = JerkHumanoid(penalize_jerk=config.penalize_jerk, jerk_weight=config.jerk_weight)
    else:
        try:
            env = gym.make(config.env_name)
        except:
            raise NotImplementedError

    return env
