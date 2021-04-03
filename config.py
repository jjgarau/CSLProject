class Config:

    def __init__(self):

        # Environment hyperparameters
        self.env_name = 'AntBulletEnv-v0'
        self.seed = 0
        self.render = False

        # Algorithm hyperparameters
        self.steps_per_epoch = 4000
        self.epochs = 50
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        self.train_pi_iters = 80
        self.train_v_iters = 80
        self.lam = 0.97
        self.max_ep_len = 1000
        self.target_kl = 0.01
        self.save_freq= 10

        # Logging hyperparameters
        self.verbose = True
