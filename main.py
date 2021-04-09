import os
from datetime import datetime
import gym
import pybullet as p
from environments.env_maker import make_env

from config import Config
from models.selector import RunSelector
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
	env = make_env(config)

	# Render
	if config.render:
		env.render()

	# Create logger object
	if not config.eval_mode:
		dt_string = get_datetime_string()
		dir_name = os.path.join('results', 'simulation_' + dt_string)
		os.makedirs(dir_name, exist_ok=True)
		logger = Logger(dir_name, config)
	else:
		logger = None

	# Train a model
	selector = RunSelector(config)
	selector.run(env=env, config=config, logger=logger)


if __name__ == "__main__":
	main()
