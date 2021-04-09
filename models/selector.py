import os

from models.ppo.ppo import ppo_train, ppo_eval
from models.sac.sac import sac_train, sac_eval
from models.policies import BaselinePolicy


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
            policy = self.select_policy(env, config)
            runner(policy, s)

    def select_algorithm(self, env, logger, config=None):

        config = self.config if config is None else config

        if config.algorithm == 'PPO':

            if config.eval_mode:
                model_path = os.path.join('eval', config.eval_model)

                def runner(x, y): ppo_eval(env=env, model_path=model_path, policy=x, seed=y,
                                           steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                                           max_ep_len=config.max_ep_len)
            else:
                def runner(x, y): ppo_train(env=env, policy=x, seed=y, steps_per_epoch=config.steps_per_epoch,
                                            epochs=config.epochs, gamma=config.gamma, clip_ratio=config.clip_ratio,
                                            pi_lr=config.pi_lr, vf_lr=config.vf_lr,
                                            train_pi_iters=config.train_pi_iters, train_v_iters=config.train_v_iters,
                                            lam=config.lam, max_ep_len=config.max_ep_len, target_kl=config.target_kl,
                                            save_freq=config.save_freq, logger=logger)

        elif config.algorithm == 'SAC':

            if config.eval_mode:
                model_path = os.path.join('eval', config.eval_model)

                def runner(x, y): sac_eval(env=env, model_path=model_path, seed=y,
                                           steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                                           max_ep_len=config.max_ep_len)
            else:
                def runner(x, y): sac_train(env=env, test_env=None, seed=y, steps_per_epoch=config.steps_per_epoch,
                                            epochs=config.epochs, replay_size=config.replay_size, gamma=config.gamma,
                                            polyak=config.polyak, lr=config.lr, alpha=config.alpha,
                                            batch_size=config.batch_size, start_steps=config.start_steps,
                                            update_after=config.update_after, update_every=config.update_every,
                                            num_test_episodes=config.num_test_episodes, max_ep_len=config.max_ep_len,
                                            save_freq=config.save_freq, logger=logger)

        else:
            raise NotImplementedError

        return runner

    def select_policy(self, env, config=None):

        config = self.config if config is None else config

        if config.env_type == 'Baseline':
            return BaselinePolicy(env.observation_space, env.action_space)
        else:
            raise NotImplementedError
