import gym

from environments.locomotion import JerkAnt


def make_env(config):

    if config.env_name == 'JerkAnt':
        env = JerkAnt()
    else:
        try:
            env = gym.make(config.env_name)
        except:
            raise NotImplementedError

    return env
