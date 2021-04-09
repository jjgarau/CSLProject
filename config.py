class DefaultConfig:

    def __init__(self):

        # Environment hyperparameters
        self.env_name = 'AntBulletEnv-v0'
        self.seed = 0
        self.render = False
        self.eval_mode = False
        self.eval_model = 'test.pt'
        self.policy = 'Baseline'

        # Algorithm hyperparameters
        self.algorithm = 'PPO'
        self.steps_per_epoch = 4000
        self.epochs = 51
        self.gamma = 0.99
        self.max_ep_len = 1000
        self.save_freq = 10

        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        self.train_pi_iters = 80
        self.train_v_iters = 80
        self.lam = 0.97
        self.target_kl = 0.01

        # SAC hyperparameters
        self.replay_size = int(1e6)
        self.polyak = 0.995
        self.lr = 1e-3
        self.alpha = 0.2
        self.batch_size = 100
        self.start_steps = 10000
        self.update_after = 1000
        self.update_every = 20
        self.num_test_episodes = 4

        # Logging hyperparameters
        self.verbose = True
        self.plot = True
        self.rolling = 20
        self.ci = 95


class Config(DefaultConfig):

    def __init__(self):
        super().__init__()

        self.env_name = 'JerkAnt'
