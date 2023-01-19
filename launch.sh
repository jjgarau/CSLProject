#!/bin/sh

python main.py --take_arg 1 --env_name 'MaskedAnt'
python main.py --take_arg 1 --env_name 'MaskedAnt' --mask 'body_pos'
python main.py --take_arg 1 --env_name 'MaskedAnt' --mask 'body_vel'
python main.py --take_arg 1 --env_name 'MaskedAnt' --mask 'roll'