#!/bin/sh

python main.py --take_arg 1 --env_name 'MaskedAnt' --mask 'joint_vel' --gpu_id 1
python main.py --take_arg 1 --env_name 'MaskedHumanoid' --mask 'pitch' --gpu_id 1
python main.py --take_arg 1 --env_name 'MaskedHumanoid' --mask 'joint_pos' --gpu_id 1
python main.py --take_arg 1 --env_name 'MaskedHumanoid' --mask 'joint_vel' --gpu_id 1
python main.py --take_arg 1 --env_name 'MaskedHumanoid' --mask 'contact' --gpu_id 1