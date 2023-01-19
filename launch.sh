#!/bin/sh

python main.py --take_arg 1 --env_name 'MaskedHumanoid'
python main.py --take_arg 1 --env_name 'MaskedHumanoid' --mask 'body_pos'
python main.py --take_arg 1 --env_name 'MaskedHumanoid' --mask 'body_vel'
python main.py --take_arg 1 --env_name 'MaskedHumanoid' --mask 'roll'