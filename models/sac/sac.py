from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import models.sac.core as core


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def sac_train(env, test_env=None, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000,
              epochs=100, replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100,
              start_steps=10000, update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, save_freq=1,
              logger=None):

    # Prepare logger for run
    logger.set_up_seed_episode_df(seed)
    logger.log(f'Prepare run with seed {seed}')

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Instantiate environment
    if test_env is None:
        test_env = deepcopy(env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log(f'Number of parameters: \t pi: {var_counts[0]}, \t q1: {var_counts[1]}, \t q2: {var_counts[2]}')

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from current policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q1_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # TODO: logging happens here

        # Freeze Q-networks so you don't waste computational effort computing gradients
        # for them during the policy learning step
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # TODO: logging happens here

        # Finally, update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # We use in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent():
        epoch_returns = []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = get_action(o, True)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            epoch_returns.append(ep_ret)
        logger.log_epoch(epoch_returns, epoch)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    logger.log(f'Starting epoch 0')

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        epoch = t // steps_per_epoch

        # Until start_steps have elapsed, randomly sample actions from a uniform distribution
        # for better exploration. Afterwards, use the learned policy
        a = get_action(o) if t > start_steps else env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += 1
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time horizon
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Update observation
        o = o2

        # End of trajectory handling
        if d or ep_len == max_ep_len:
            logger.log_episode(ep_ret, ep_len, epoch)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:

            # Save model
            if epoch % save_freq == 0 or epoch == epochs:
                logger.save_model(ac, epoch)

            test_agent()

            logger.log(f'Starting epoch {epoch + 1}')

            # TODO: A lot of logging happens here

    logger.save_run()
    logger.log('\n\n')


def sac_eval(env, model_path, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000,
             epochs=100, max_ep_len=1000):

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(model_path))
    ac.eval()

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop
    for t in range(total_steps):

        epoch = t // steps_per_epoch

        # Get action deterministically
        a = get_action(o, True)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += 1
        ep_len += 1

        # Update observation
        o = o2

        # End of trajectory handling
        if d or ep_len == max_ep_len:
            o, ep_ret, ep_len = env.reset(), 0, 0
