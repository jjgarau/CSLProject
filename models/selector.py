import os
import itertools

from models.ppo.ppo import ppo_train, ppo_eval
from models.sac.sac import sac_train, sac_eval
from models.policies import BaselinePolicy, PreviousActionPolicy, ActionDifferencePolicy, MovingAveragePolicy,\
    RecurrentPolicy
from environments.env_maker import make_env


class RunSelector:

    def __init__(self, config):

        self.config = config

    def run(self, logger=None, config=None):

        config = self.config if config is None else config

        seed = config.seed
        if isinstance(seed, float):
            seed = int(seed)
        if isinstance(seed, int):
            seed = [seed]
        assert type(seed) is list

        policy = config.policy
        if isinstance(policy, str):
            policy = [policy]
        assert type(policy) is list

        runner = self.select_algorithm(logger, config)

        for p, s in itertools.product(policy, seed):

            # Create environment
            env = make_env(config)

            # Render
            if config.render:
                env.render()

            pol = self.select_policy(env, p, config)
            runner(pol, s, env)

    def select_algorithm(self, logger, config=None):

        config = self.config if config is None else config

        if config.algorithm == 'PPO':

            if config.eval_mode:
                model_path = os.path.join('eval', config.eval_model)

                def runner(x, y, env): ppo_eval(env=env, model_path=model_path, policy=x, seed=y,
                                                steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                                                max_ep_len=config.max_ep_len)
            else:
                load_model_path = None if config.train_from_scratch else os.path.join('eval', config.load_model_path)

                def runner(x, y, env): ppo_train(env=env, policy=x, seed=y, steps_per_epoch=config.steps_per_epoch,
                                                 epochs=config.epochs, gamma=config.gamma, clip_ratio=config.clip_ratio,
                                                 pi_lr=config.pi_lr, vf_lr=config.vf_lr,
                                                 train_pi_iters=config.train_pi_iters,
                                                 train_v_iters=config.train_v_iters, lam=config.lam,
                                                 max_ep_len=config.max_ep_len, target_kl=config.target_kl,
                                                 save_freq=config.save_freq, logger=logger, gpu=config.gpu,
                                                 load_model_path=load_model_path)

        elif config.algorithm == 'SAC':

            if config.eval_mode:
                model_path = os.path.join('eval', config.eval_model)

                def runner(x, y, env): sac_eval(env=env, model_path=model_path, seed=y,
                                                steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                                                max_ep_len=config.max_ep_len)
            else:
                def runner(x, y, env): sac_train(env=env, test_env=None, seed=y, steps_per_epoch=config.steps_per_epoch,
                                                 epochs=config.epochs, replay_size=config.replay_size,
                                                 gamma=config.gamma, polyak=config.polyak, lr=config.lr,
                                                 alpha=config.alpha, batch_size=config.batch_size,
                                                 start_steps=config.start_steps, update_after=config.update_after,
                                                 update_every=config.update_every,
                                                 num_test_episodes=config.num_test_episodes,
                                                 max_ep_len=config.max_ep_len, save_freq=config.save_freq,
                                                 logger=logger)

        else:
            raise NotImplementedError

        return runner

    def select_policy(self, env, p=None, config=None):

        config = self.config if config is None else config
        p = config.policy if p is None else p

        if p == 'Baseline':
            return BaselinePolicy(env.observation_space, env.action_space)
        elif p == 'Moving average':
            return MovingAveragePolicy(env.observation_space, env.action_space, window_size=config.window_size)
        elif p == 'Action difference':
            return ActionDifferencePolicy(env.observation_space, env.action_space)
        elif p == 'Previous action':
            return PreviousActionPolicy(env.observation_space, env.action_space)
        elif p == 'Recurrent':
            return RecurrentPolicy(env.observation_space, env.action_space)
        else:
            raise NotImplementedError
