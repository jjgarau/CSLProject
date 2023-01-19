import gym

from environments.locomotion import JerkAnt, JerkHumanoid, JerkHalfCheetah, JerkHopper
from environments.locomotion import MaskedAnt


def make_env(config):

    if config.env_name == 'JerkAnt':
        env = JerkAnt(penalize_jerk=config.penalize_jerk, jerk_weight=config.jerk_weight)
    elif config.env_name == 'JerkHumanoid':
        env = JerkHumanoid(penalize_jerk=config.penalize_jerk, jerk_weight=config.jerk_weight)
    elif config.env_name == 'JerkHalfCheetah':
        env = JerkHalfCheetah(penalize_jerk=config.penalize_jerk, jerk_weight=config.jerk_weight)
    elif config.env_name == 'JerkHopper':
        env = JerkHopper(penalize_jerk=config.penalize_jerk, jerk_weight=config.jerk_weight)
    elif config.env_name == 'MaskedAnt':
        env = MaskedAnt(mask=config.mask)
    else:
        try:
            env = gym.make(config.env_name)
        except:
            raise NotImplementedError

    return env
