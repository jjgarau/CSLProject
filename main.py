import os
import time
from datetime import datetime
import gym
import pybullet_envs
import pybullet as p

from config import Config
from models.ppo import ppo_train, ppo_eval
from models.logger import Logger


from pybullet_envs.gym_locomotion_envs import AntBulletEnv


def get_datetime_string():
	now = datetime.now()
	dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
	return dt_string


def main():

	# Load config parameters
	config = Config()

	# Connect to PyBullet
	p.connect(p.DIRECT)

	# Create environment
	env = gym.make(config.env_name)

	# Render
	if config.render:
		env.render()

	# Create logger object
	# TODO: this only if we're in train mode
	if not config.eval_mode:
		dt_string = get_datetime_string()
		dir_name = os.path.join('results', 'simulation_' + dt_string)
		os.makedirs(dir_name, exist_ok=True)
		logger = Logger(dir_name, config)
	else:
		logger = None

	# Train a model
	# TODO: eventually send this to a selector
	if config.eval_mode:
		model_path = os.path.join('eval', config.eval_model)
		ppo_eval(env=env, model_path=model_path, seed=config.seed, steps_per_epoch=config.steps_per_epoch,
				 epochs=config.epochs, max_ep_len=config.max_ep_len)
	else:
		ppo_train(env=env, seed=config.seed, steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
				  gamma=config.gamma, clip_ratio=config.clip_ratio, pi_lr=config.pi_lr, vf_lr=config.vf_lr,
				  train_pi_iters=config.train_pi_iters, train_v_iters=config.train_v_iters, lam=config.lam,
				  max_ep_len=config.max_ep_len, target_kl=config.target_kl, save_freq=config.save_freq, logger=logger)


if __name__ == "__main__":
	main()
