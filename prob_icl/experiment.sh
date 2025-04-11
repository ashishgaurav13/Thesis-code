#!/usr/bin/env sh

mkdir -p "output/gridworld"
mkdir -p "output/cartpole"
mkdir -p "output/mujoco_ant"
mkdir -p "output/mujoco_hc"
mkdir -p "output/highd"

# Expert data generation

python3 -B prob_icl.py -env gridworld -o output/gridworld/expert -seed 0 -beta 0.99 -delta 0.6 -expert_only
python3 -B prob_icl.py -env cartpole -o output/cartpole/expert -seed 0 -beta 20 -delta 0.7 -expert_only
python3 -B prob_icl.py -env mujoco_ant -o output/mujoco_ant/expert -seed 0 -beta 15 -delta 0.7 -expert_only
python3 -B prob_icl.py -env mujoco_hc -o output/mujoco_hc/expert -seed 0 -beta 15 -delta 0.5 -expert_only

# ICL

python3 -B icl.py -env gridworld -o output/gridworld/icl -seed 1 -expert_dir output/gridworld/expert
python3 -B icl.py -env cartpole -o output/cartpole/icl -seed 1 -expert_dir output/cartpole/expert -ppo_lag
python3 -B icl.py -env mujoco_ant -o output/mujoco_ant/icl -seed 1 -expert_dir output/mujoco_ant/expert -ppo_lag
python3 -B icl.py -env mujoco_hc -o output/mujoco_hc/icl -seed 1 -expert_dir output/mujoco_hc/expert -ppo_lag
python3 -B icl.py -env highd -o output/highd/icl -seed 1 -beta 0.1

# ICL w/ beta+log(1-delta)

python3 -B icl.py -env gridworld -o output/gridworld/iclbetadelta -seed 2 -expert_dir output/gridworld/expert -beta 0.07
python3 -B icl.py -env cartpole -o output/cartpole/iclbetadelta -seed 2 -expert_dir output/cartpole/expert -beta 18.79 -ppo_lag
python3 -B icl.py -env mujoco_ant -o output/mujoco_ant/iclbetadelta -seed 2 -expert_dir output/mujoco_ant/expert -beta 13.79 -ppo_lag
python3 -B icl.py -env mujoco_hc -o output/mujoco_hc/iclbetadelta -seed 2 -expert_dir output/mujoco_hc/expert -beta 14.30 -ppo_lag

# IPCL

python3 -B prob_icl.py -env gridworld -o output/gridworld/ipcl -seed 3 -beta 0.99 -delta 0.3 -expert_dir output/gridworld/expert
python3 -B prob_icl.py -env cartpole -o output/cartpole/ipcl -seed 3 -beta 20 -delta 0.7 -expert_dir output/cartpole/expert
python3 -B prob_icl.py -env mujoco_ant -o output/mujoco_ant/ipcl -seed 3 -beta 15 -delta 0.7 -expert_dir output/mujoco_ant/expert
python3 -B prob_icl.py -env mujoco_hc -o output/mujoco_hc/ipcl -seed 3 -beta 15 -delta 0.5 -expert_dir output/mujoco_hc/expert
python3 -B prob_icl.py -env highd -o output/highd/ipcl -seed 1 -beta 0.1 -delta 0.5
python3 -B prob_icl.py -env highd -o output/highd/ipcl -seed 1 -beta 0.1 -delta 0.9



