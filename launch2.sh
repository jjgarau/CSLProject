#!/bin/sh

python main.py --take_arg 1 --env_name 'MaskedHopper' --mask 'pitch' --gpu_id 1
python main.py --take_arg 1 --env_name 'MaskedHopper' --mask 'joint_pos' --gpu_id 1
python main.py --take_arg 1 --env_name 'MaskedHopper' --mask 'joint_vel' --gpu_id 1
python main.py --take_arg 1 --env_name 'MaskedHopper' --mask 'contact' --gpu_id 1