class DefaultConfig:

    def __init__(self):

        # Environment hyperparameters
        self.env_name = 'AntBulletEnv-v0'
        self.seed = 0
        self.render = False
        self.eval_mode = False
        self.eval_model = 'test.pt'
        self.penalize_jerk = False
        self.jerk_weight = 50

        # Policy hyperparameters
        self.policy = 'Baseline'
        self.window_size = 10
        self.gpu = True
        self.train_from_scratch = True
        self.load_model_path = 'train.pt'
        self.recurrent_hidden_size = 64
        self.recurrent_layers = 2

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
        self.policy = ['Baseline', 'Moving average', 'Previous action', 'Action difference', 'Recurrent']
        self.gpu = False
        # self.penalize_jerk = True
        self.seed = [10, 142, 1100, 112313, 112423423]
        self.epochs = 201
        self.save_freq = 50

