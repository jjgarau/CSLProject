import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete

from models.ppo.core import MLPActorCritic, MLPCritic, MLPGaussianActor, MLPCategoricalActor
from pybullet_envs.gym_locomotion_envs import AntBulletEnv
from pybullet_envs.robot_locomotors import Ant
from pybullet_envs.robot_bases import MJCFBasedRobot


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

    def send_to_device(self, device):
        raise NotImplementedError

    def new_episode(self):
        pass


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

    def send_to_device(self, device):
        self.ac.to(device)

    def load_model(self, model_path, eval_model=True):
        self.ac.load_state_dict(torch.load(model_path))
        if eval_model:
            self.ac.eval()


class MovingAveragePolicy(BaselinePolicy):

    def __init__(self, observation_space, action_space, window_size=10, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__(observation_space, action_space, hidden_sizes, activation)
        self.window_size = window_size
        self.action_log = []

    def get_name(self):
        return "Moving average"

    def step(self, obs):
        a, v, logp = self.ac.step(obs)
        self.action_log.append(a)
        if len(self.action_log) > self.window_size:
            self.action_log = self.action_log[-self.window_size:]
        a_act = np.mean(self.action_log, axis=0)
        return (a, a_act), v, logp

    def new_episode(self):
        self.action_log = []


class ActionDifferencePolicy(BaselinePolicy):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__(observation_space, action_space, hidden_sizes, activation)
        self.action_space = action_space
        self.action = np.zeros(self.action_space.shape, dtype=float)

    def get_name(self):
        return "Action difference"

    def step(self, obs):
        a, v, logp = self.ac.step(obs)
        self.action += a
        return (a, self.action), v, logp

    def new_episode(self):
        self.action = np.zeros(self.action_space.shape, dtype=float)


class PreviousActionPolicy(Policy):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        if isinstance(action_space, Box):
            self.actor = MLPGaussianActor(self.obs_dim + self.act_dim, self.act_dim, hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.actor = MLPCategoricalActor(self.obs_dim + self.act_dim, self.act_dim, hidden_sizes, activation)
        else:
            self.actor = None

        self.critic = MLPCritic(self.obs_dim, hidden_sizes, activation)

        self.previous_action = torch.zeros(self.act_dim, dtype=torch.float32)

    def get_name(self):
        return "Previous action"

    def pi(self, obs, act):
        shifted_act = torch.cat((torch.zeros((1, act.shape[-1])), act))
        shifted_act = shifted_act[:-1]
        full_obs = torch.cat((obs, shifted_act), dim=-1)
        return self.actor(full_obs, act)

    def v(self, obs):
        return self.critic(obs)

    def pi_params(self):
        return self.actor.parameters()

    def v_params(self):
        return self.critic.parameters()

    def step(self, obs):
        with torch.no_grad():
            full_obs = torch.cat((obs, self.previous_action))
            pi = self.actor._distribution(full_obs)
            a = pi.sample()
            logp_a = self.actor._log_prob_from_distribution(pi, a)
            v = self.critic(obs)
            self.previous_action = a
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def get_pi(self):
        return self.actor

    def get_v(self):
        return self.critic

    def get_model(self):
        return self.actor, self.critic

    def load_model(self, model_path, eval_model=True):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['0'])
        self.critic.load_state_dict(checkpoint['1'])
        if eval_model:
            self.actor.eval()
            self.critic.eval()

    def send_to_device(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def new_episode(self):
        self.previous_action = torch.zeros(self.act_dim, dtype=torch.float32)


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

    def send_to_device(self, device):
        pass

    def load_model(self, model_path):
        pass
