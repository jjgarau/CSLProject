import numpy as np

from pybullet_envs.gym_locomotion_envs import AntBulletEnv


class JerkAnt(AntBulletEnv):

    def __init__(self):
        super().__init__()
        self.obs_sequence = []

    def reset(self):
        obs = super().reset()
        self.obs_sequence.append(obs)
        return obs

    def step(self, a):
        obs, r, d, _ = super().step(a)
        self.obs_sequence.append(obs)
        return obs, r, d, None

    def get_jerk_from_v(self, v):
        a = np.diff(v, axis=0)
        j = np.diff(a, axis=0)
        j = np.max(np.sqrt(np.sum(j ** 2, axis=-1)))
        return j

    def compute_jerk_body(self, seq):
        v = seq[:, 3:6]
        return self.get_jerk_from_v(v)

    def compute_jerk_joints(self, seq):
        seq = [seq[:, 9 + 2 * i] for i in range(8)]
        v = np.array(seq).T
        return self.get_jerk_from_v(v)

    def compute_jerk(self):
        seq = np.array(self.obs_sequence)
        body = self.compute_jerk_body(seq)
        joints = self.compute_jerk_joints(seq)
        return body, joints
