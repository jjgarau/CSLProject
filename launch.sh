#!/bin/sh

python main.py --take_arg 1 --env_name 'MaskedHopper'
python main.py --take_arg 1 --env_name 'MaskedHopper' --mask 'body_pos'
python main.py --take_arg 1 --env_name 'MaskedHopper' --mask 'body_vel'
python main.py --take_arg 1 --env_name 'MaskedHopper' --mask 'roll'