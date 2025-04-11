#!/bin/bash

cd interface

python train_icrl_new.py ../config/Other/gridworld.yaml -n 5 -s 1

python train_icrl_new.py ../config/Other/cartpole.yaml -n 5 -s 2

python train_icrl_new.py ../config/Other/ant.yaml -n 5 -s 3

python train_icrl_new.py ../config/Other/hc.yaml -n 5 -s 4
