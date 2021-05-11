import numpy as np

from pybullet_envs.gym_locomotion_envs import AntBulletEnv, HumanoidBulletEnv, HalfCheetahBulletEnv


class JerkAnt(AntBulletEnv):

    def __init__(self, penalize_jerk=False, jerk_weight=50):
        super().__init__()
        self.obs_sequence = []
        self.penalize_jerk = penalize_jerk
        self.jerk_weight = jerk_weight

    def reset(self):
        obs = super().reset()
        self.obs_sequence = []
        self.obs_sequence.append(obs)
        return obs

    def step(self, a):
        obs, r, d, debug = super().step(a)
        self.obs_sequence.append(obs)
        r = max(r, 0)
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
        v = seq[:, 3:6]
        return self.get_jerk_from_v(v)

    def compute_jerk_joints(self, seq):
        seq = [seq[:, 9 + 2 * i] for i in range(8)]
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


class JerkHumanoid(HumanoidBulletEnv):

    def __init__(self, penalize_jerk=False, jerk_weight=50):
        super().__init__()
        self.obs_sequence = []
        self.penalize_jerk = penalize_jerk
        self.jerk_weight = jerk_weight

    def reset(self):
        obs = super().reset()
        self.obs_sequence = []
        self.obs_sequence.append(obs)
        return obs

    def step(self, a):
        obs, r, d, debug = super().step(a)
        self.obs_sequence.append(obs)
        r = max(r, -30)
        if self.penalize_jerk:
            j = 0  # TODO: Fix this
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
        pass

    def compute_jerk_joints(self, seq):
        pass

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
