# Uncertainty-aware Inverse Constrained Reinforcement Learning
![](https://github.com/Jasonxu1225/Uncertainty-aware-Inverse-Constrained-Reinforcement-Learning/blob/main/workflow.jpg)
This is the code for the paper [Uncertainty-aware Constraint Inference in Inverse Constrained Reinforcement Learning](https://openreview.net/pdf?id=ILYjDvUM6U) published at ICLR 2024. This code is taken from [github.com/Jasonxu1225/Uncertainty-aware-Inverse-Constrained-Reinforcement-Learning](https://github.com/Jasonxu1225/Uncertainty-aware-Inverse-Constrained-Reinforcement-Learning).

## Create Python Environment 
1. Please install the conda before proceeding.
2. Create a conda environment and install the packages:
   
```
mkdir save_model
mkdir evaluate_model
conda env create -n cn39 python=3.9 -f python_environment.yml
conda activate cn39
```
You can also first install Python 3.9 with the torch (2.0.1+cu117) and then install the packages listed in `python_environment.yml`.

## Setup MuJoCo Environment
1. Download the MuJoCo version 2.1 binaries for Linux.
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
3. Install and use mujoco-py.
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment

export MUJOCO_PY_MUJOCO_PATH=YOUR_MUJOCO_DIR/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_MUJOCO_DIR/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## Setup tools

Change directory into `tools_dir` and run `pip install -e .`.

## Generate Expert Demonstration

Expert demonstrations are present in `data/expert_data` folder.

## Train ICRL Algorithms

See `experiment.sh`.

## References

[Uncertainty-aware Constraint Inference in Inverse Constrained Reinforcement Learning, Xu & Liu (2024)](https://openreview.net/forum?id=ILYjDvUM6U)