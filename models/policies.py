import numpy as np
import torch
import torch.nn as nn

from models.ppo.core import MLPActorCritic


class Policy:

    def get_name(self):
        raise NotImplementedError

    def pi(self, obs, act):
        raise NotImplementedError

    def v(self, obs):
        raise NotImplementedError

    def pi_params(self):
        raise NotImplementedError

    def v_params(self):
        raise NotImplementedError

    def get_pi(self):
        raise NotImplementedError

    def get_v(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError


class BaselinePolicy(Policy):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        self.ac = MLPActorCritic(observation_space, action_space, hidden_sizes, activation)

    def get_name(self):
        return "Baseline"

    def pi(self, obs, act):
        return self.ac.pi(obs, act)

    def v(self, obs):
        return self.ac.v(obs)

    def pi_params(self):
        return self.ac.pi.parameters()

    def v_params(self):
        return self.ac.v.parameters()

    def step(self, obs):
        return self.ac.step(obs)

    def get_pi(self):
        return self.ac.pi

    def get_v(self):
        return self.ac.v

    def get_model(self):
        return self.ac

    def load_model(self, model_path):
        self.ac.load_state_dict(torch.load(model_path))
        self.ac.eval()


class PreviousActionPolicy(Policy):

    def __init__(self):
        pass

    def get_name(self):
        return "Previous action"

    def pi(self, obs, act):
        pass

    def v(self, obs):
        pass

    def pi_params(self):
        pass

    def v_params(self):
        pass

    def step(self, obs):
        pass

    def get_pi(self):
        pass

    def get_v(self):
        pass

    def get_model(self):
        pass

    def load_model(self, model_path):
        pass


class RecurrentPolicy(Policy):

    def __init__(self):
        pass

    def get_name(self):
        return "Recurrent"

    def pi(self, obs, act):
        pass

    def v(self, obs):
        pass

    def pi_params(self):
        pass

    def v_params(self):
        pass

    def step(self, obs):
        pass

    def get_pi(self):
        pass

    def get_v(self):
        pass

    def get_model(self):
        pass

    def load_model(self, model_path):
        pass


class RollingAveragePolicy(Policy):

    def __init__(self):
        pass

    def get_name(self):
        return "Rolling average"

    def pi(self, obs, act):
        pass

    def v(self, obs):
        pass

    def pi_params(self):
        pass

    def v_params(self):
        pass

    def step(self, obs):
        pass

    def get_pi(self):
        pass

    def get_v(self):
        pass

    def get_model(self):
        pass

    def load_model(self, model_path):
        pass


class ActionDifferencePolicy(Policy):

    def __init__(self):
        pass

    def get_name(self):
        return "Action difference"

    def pi(self, obs, act):
        pass

    def v(self, obs):
        pass

    def pi_params(self):
        pass

    def v_params(self):
        pass

    def step(self, obs):
        pass

    def get_pi(self):
        pass

    def get_v(self):
        pass

    def get_model(self):
        pass

    def load_model(self, model_path):
        pass
