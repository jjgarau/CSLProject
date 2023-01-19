import numpy as np

from pybullet_envs.gym_locomotion_envs import AntBulletEnv, HumanoidBulletEnv, HalfCheetahBulletEnv, HopperBulletEnv


class JerkEnv:

    def __init__(self, body_vel_slice, num_actions, penalize_jerk, jerk_weight, reward_lb=0):
        self.obs_sequence = []
        self.robot = None

        self.body_vel_slice = body_vel_slice
        self.num_actions = num_actions
        self.penalize_jerk = penalize_jerk
        self.jerk_weight = jerk_weight
        self.reward_lb = reward_lb

    @property
    def observation_space(self):
        return self.robot.observation_space

    @property
    def action_space(self):
        return self.robot.action_space

    def seed(self, seed):
        self.robot.seed(seed)

    def render(self):
        self.robot.render()

    def reset(self):
        obs = self.robot.reset()
        self.obs_sequence = []
        self.obs_sequence.append(obs)
        return obs

    def step(self, a):
        obs, r, d, debug = self.robot.step(a)
        self.obs_sequence.append(obs)
        r = max(r, self.reward_lb)
        if self.penalize_jerk:
            j = sum(self.compute_jerk(all_t=False))
            r_jerk = r - j * self.jerk_weight
            return obs, (r, r_jerk), d, debug
        else:
            return obs, r, d, debug

    def get_jerk_from_v(self, v, take_norm=True):
        a = np.diff(v, axis=0)
        j = np.diff(a, axis=0)
        if take_norm:
            j = np.max(np.sqrt(np.sum(j ** 2, axis=-1)))
        else:
            j = np.max(np.abs(j))
        return j

    def compute_jerk_body(self, seq):
        v = seq[:, self.body_vel_slice]
        return self.get_jerk_from_v(v)

    def compute_jerk_joints(self, seq):
        seq = [seq[:, 9 + 2 * i] for i in range(self.num_actions)]
        v = np.array(seq).T
        return self.get_jerk_from_v(v, take_norm=False)

    def compute_jerk(self, all_t=True):
        if len(self.obs_sequence) <= 2:
            return 0, 0
        if all_t:
            seq = np.array(self.obs_sequence)
        else:
            seq = np.array(self.obs_sequence[-3:])
        body = self.compute_jerk_body(seq)
        joints = self.compute_jerk_joints(seq)
        return body, joints


class JerkAnt(JerkEnv):

    def __init__(self, penalize_jerk=False, jerk_weight=50):
        super().__init__(body_vel_slice=slice(3, 6), num_actions=8, penalize_jerk=penalize_jerk,
                         jerk_weight=jerk_weight, reward_lb=0)
        self.robot = AntBulletEnv()


class JerkHumanoid(JerkEnv):

    def __init__(self, penalize_jerk=False, jerk_weight=50):
        super().__init__(body_vel_slice=slice(3, 6), num_actions=17, penalize_jerk=penalize_jerk,
                         jerk_weight=jerk_weight, reward_lb=-30)
        self.robot = HumanoidBulletEnv()


class JerkHalfCheetah(JerkEnv):

    def __init__(self, penalize_jerk=False, jerk_weight=50):
        super().__init__(body_vel_slice=slice(3, 6), num_actions=6, penalize_jerk=penalize_jerk,
                         jerk_weight=jerk_weight, reward_lb=-100)
        self.robot = HalfCheetahBulletEnv()


class JerkHopper(JerkEnv):

    def __init__(self, penalize_jerk=False, jerk_weight=50):
        super().__init__(body_vel_slice=slice(3, 6), num_actions=3, penalize_jerk=penalize_jerk,
                         jerk_weight=jerk_weight, reward_lb=0)
        self.robot = HopperBulletEnv()


class MaskedAnt(JerkAnt):

    def __init__(self, mask=None):
        super().__init__()
        self.mask = mask

    def tune_obs(self, obs):
        if self.mask is None:
            return obs
        elif self.mask == 'body_pos':
            obs[:3] = 0
        elif self.mask == 'body_vel':
            obs[3:6] = 0
        elif self.mask == 'roll':
            obs[6] = 0
        elif self.mask == 'pitch':
            obs[7] = 0
        elif self.mask == 'joint_pos':
            for i in range(self.num_actions):
                obs[8 + 2 * i] = 0
        elif self.mask == 'joint_vel':
            for i in range(self.num_actions):
                obs[8 + 2 * 1 + 1] = 0
        elif self.mask == 'contact':
            obs[24:] = 0
        return obs

    def reset(self):
        obs = super().reset()
        return self.tune_obs(obs)

    def step(self, a):
        obs, r, d, debug = super().step(a)
        obs = self.tune_obs(obs)
        return obs, r, d, debug
