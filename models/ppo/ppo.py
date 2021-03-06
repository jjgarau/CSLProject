import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import time
import models.ppo.core as core


class PPOBuffer:

    def __init__(self, obs_dim, act_dim, h_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        if h_dim is not None:
            self.h_pi_buf = np.zeros(core.combined_shape(size, h_dim), dtype=np.float32)
            self.h_v_buf = np.zeros(core.combined_shape(size, h_dim), dtype=np.float32)
            self.store_h = True
        else:
            self.h_pi_buf, self.h_v_buf = None, None
            self.store_h = False
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, done, h_pi=None, h_v=None):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        if self.store_h:
            self.h_pi_buf[self.ptr] = h_pi
            self.h_v_buf[self.ptr] = h_v
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf,
                    done=self.done_buf)
        if self.store_h:
            data['h_pi'] = self.h_pi_buf
            data['h_v'] = self.h_v_buf
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo_train(env, policy, seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
              vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01,
              save_freq=10, logger=None, gpu=False, load_model_path=None, recurrent=False, hidden_size=64, gpu_id='0'):

    # Prepare logger for run
    logger.set_up_seed_episode_df(policy, seed)
    logger.log(f'Prepare run with policy {policy.get_name()} and seed {seed}')

    # Set up device
    if gpu and torch.cuda.is_available():
        device = 'cuda:' + gpu_id
    else:
        device = 'cpu'

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Start training from previous model
    if load_model_path is not None:
        policy.load_model(load_model_path)

    # Instantiate environment
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [policy.get_pi(), policy.get_v()])
    logger.log(f'Number of parameters: \t pi: {var_counts[0]}, \t v: {var_counts[1]}')

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    h_dim = hidden_size if recurrent else None
    buf = PPOBuffer(obs_dim, act_dim, h_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs, act, adv, logp_old = obs.to(device), act.to(device), adv.to(device), logp_old.to(device)
        hidden_pi = data['h_pi'].to(device) if recurrent else None

        # Policy loss
        if recurrent:
            pi, logp = policy.pi(obs, act, hidden_pi)
        else:
            pi, logp = policy.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = - (torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        obs, ret = obs.to(device), ret.to(device)
        hidden_v = data['h_v'].to(device) if recurrent else None

        if recurrent:
            return ((policy.v(obs, hidden_v) - ret) ** 2).mean()
        else:
            return ((policy.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(policy.pi_params(), lr=pi_lr)
    vf_optimizer = Adam(policy.v_params(), lr=vf_lr)

    def update():
        policy.send_to_device(device)
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0
    policy.new_episode()

    for epoch in tqdm(range(epochs), desc="Epoch progress"):

        logger.log(f'Starting epoch {epoch}')
        policy.send_to_device('cpu')

        for t in range(local_steps_per_epoch):

            # Consider hidden state if recurrent policy
            if recurrent:
                a, v, logp, h_pi, h_v = policy.step(torch.as_tensor(o, dtype=torch.float32))
            else:
                a, v, logp = policy.step(torch.as_tensor(o, dtype=torch.float32))
                h_pi, h_v = None, None

            # Consider applied action and action to train on separately
            if type(a) is list or type(a) is tuple:
                a_train, a_act = a
            else:
                a_train, a_act = a, a

            next_o, r, d, _ = env.step(a_act)

            # Consider logged reward and policy reward separately
            if type(r) is list or type(r) is tuple:
                r_log, r_train = r
            else:
                r_log, r_train = r, r

            ep_ret += r_log
            ep_len += 1

            # save and log
            buf.store(o, a_train, r_train, v, logp, d, h_pi, h_v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:

                if terminal:
                    logger.log_episode(ep_ret, ep_len, epoch, env)

                if epoch_ended:
                    logger.log_epoch(epoch)
                    if not terminal:
                        logger.log(f'Warning: trajectory cut off by epoch at {ep_len} steps')

                if timeout or epoch_ended:
                    if recurrent:
                        _, v, _, _, _ = policy.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        _, v, _ = policy.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0

                buf.finish_path(v)
                o, ep_ret, ep_len = env.reset(), 0, 0
                policy.new_episode()

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_model(policy.get_model(), epoch)

        # Perform PPO update
        update()

    logger.save_run()
    logger.log('\n\n')


def ppo_eval(env, model_path, policy, seed=0, steps_per_epoch=4000, epochs=50, max_ep_len=1000):

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Create actor-critic module
    policy.load_model(model_path)

    # Prepare for interaction with the environment
    local_steps_per_epoch = steps_per_epoch

    o, ep_ret, ep_len = env.reset(), 0, 0
    policy.new_episode()

    for _ in tqdm(range(epochs), desc="Epoch progress"):

        for t in range(local_steps_per_epoch):

            a, v, logp = policy.step(torch.as_tensor(o, dtype=torch.float32))

            if type(a) is list or type(a) is tuple:
                a = a[1]

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:

                if terminal:
                    print(f'Episode return: {ep_ret}')
                if epoch_ended and not terminal:
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps')

                o, ep_ret, ep_len = env.reset(), 0, 0
                policy.new_episode()
