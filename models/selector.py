import os

from models.ppo.ppo import ppo_train, ppo_eval
from models.sac.sac import sac_train, sac_eval


class RunSelector:

    def __init__(self, config):

        self.config = config

    def run(self, env, logger=None, config=None):

        config = self.config if config is None else config

        seed = config.seed
        if isinstance(seed, float):
            seed = int(seed)
        if isinstance(seed, int):
            seed = [seed]
        assert type(seed) is list

        runner = self.select_algorithm(env, logger, config)

        for s in seed:
            runner(s)

    def select_algorithm(self, env, logger, config=None):

        config = self.config if config is None else config

        if config.algorithm == 'PPO':

            if config.eval_mode:
                model_path = os.path.join('eval', config.eval_model)
                def runner(x): ppo_eval(env=env, model_path=model_path, seed=x, steps_per_epoch=config.steps_per_epoch,
                                        epochs=config.epochs, max_ep_len=config.max_ep_len)
            else:
                def runner(x): ppo_train(env=env, seed=x, steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                                         gamma=config.gamma, clip_ratio=config.clip_ratio, pi_lr=config.pi_lr,
                                         vf_lr=config.vf_lr, train_pi_iters=config.train_pi_iters,
                                         train_v_iters=config.train_v_iters, lam=config.lam,
                                         max_ep_len=config.max_ep_len, target_kl=config.target_kl,
                                         save_freq=config.save_freq, logger=logger)

        elif config.algorithm == 'SAC':

            if config.eval_mode:
                pass
            else:
                pass

        else:
            raise NotImplementedError

        return runner
