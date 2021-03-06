import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = self._log_prob_from_distribution(pi, act) if act is not None else None
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = - 0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        else:
            self.pi = None

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]


class RNNActor(nn.Module):

    def _distribution(self, obs, h):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, h, act=None):
        pi, _ = self._distribution(obs, h)
        logp_a = self._log_prob_from_distribution(pi, act) if act is not None else None
        return pi, logp_a


class RNNCategoricalActor(RNNActor):

    def __init__(self, obs_dim, act_dim, hidden_size, num_layers):
        super().__init__()
        self.logits_net = nn.GRU(input_size=obs_dim, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, act_dim)

    def _distribution(self, obs, h):
        logits, h = self.logits_net(obs, h)
        logits = self.linear(logits).squeeze()
        return Categorical(logits=logits), h

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class RNNGaussianActor(RNNActor):

    def __init__(self, obs_dim, act_dim, hidden_size, num_layers):
        super().__init__()
        log_std = - 0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.GRU(input_size=obs_dim, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, act_dim)

    def _distribution(self, obs, h):
        mu, h = self.mu_net(obs, h)
        mu = self.linear(mu).squeeze()
        std = torch.exp(self.log_std)
        return Normal(mu, std), h

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class RNNCritic(nn.Module):

    def __init__(self, obs_dim, hidden_size, num_layers=1):
        super().__init__()
        self.v_net = nn.GRU(input_size=obs_dim, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, obs, h):
        v, h = self.v_net(obs, h)
        v = self.linear(v).squeeze()
        return v, h


class RNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_size=64, num_layers=1):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = RNNGaussianActor(obs_dim, action_space.shape[0], hidden_size, num_layers)
        elif isinstance(action_space, Discrete):
            self.pi = RNNCategoricalActor(obs_dim, action_space.shape[0], hidden_size, num_layers)
        else:
            self.pi = None

        self.v = RNNCritic(obs_dim, hidden_size, num_layers)

    def step(self, obs, h_pi, h_v):
        with torch.no_grad():
            pi, new_h_pi = self.pi._distribution(obs, h_pi)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v, new_h_v = self.v(obs, h_v)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), new_h_pi, new_h_v
