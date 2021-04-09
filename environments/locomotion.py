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

    def compute_jerk(self):
        return 0
