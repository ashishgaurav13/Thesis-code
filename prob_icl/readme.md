## Inverse Constraint Learning

This repository contains the code for the probabilistic ICL paper.

The `uaicrl` folder contains benchmark code for UAICRL baseline. 

## Setup

* Setup [miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
* `conda create -n my_env`
* `conda activate my_env`
* `conda install -c conda-forge python=3.11 libarchive mamba`
* `mamba update --all`
* `mamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`
* Install following packages or equivalent: `apt install freeglut3-dev libosmesa6-dev patchelf xvfb libglew-dev build-essential`
* `mamba install crlibm tensorflow-gpu==2.15.0 tensorflow-probability==0.23.0`
* `mamba install -c conda-forge mpi4py`
* `pip install cython<3 gym<0.26.0 gymnasium pynvml mkl==2023.2.0`
* Install Mujoco 2.1.0
* Install `tools` package by running `pip install .` in the root directory. 
* Install `mujoco_environments` package by running `pip install -e .` in the `mujoco_environments` directory.

## High level workflow

* If you face any OpenGL error, install `Xvfb` and prefix the command with `xvfb-run -a`.
* Command examples are given in `experiment.sh` (provided for one seed, may need to run several times with different seeds).

## Credits

Please check the individual repositories for licenses.
* OpenAI safety agents (`tools.safe_rl`):
  * https://github.com/openai/safety-starter-agents
* HighD dataset
  * https://www.highd-dataset.com
  * We include one sample set of assets (#17) from the dataset in the code, since it is necessary to run the HighD environment.
* Wise-Move environment
  * https://git.uwaterloo.ca/wise-lab/wise-move
  * https://github.com/ashishgaurav13/wm2
* Gridworld environment
  * https://github.com/yrlu/irl-imitation
* Gym environment and wrappers
  * https://github.com/ikostrikov/pytorch-ddpg-naf
  * https://github.com/vwxyzjn/cleanrl
  * https://github.com/openai/gym
* Normalizing flows
  * https://github.com/ikostrikov/pytorch-flows
  * https://github.com/tonyduan/normalizing-flows
  * https://github.com/VincentStimper/normalizing-flows
* UAICRL
  * https://github.com/Jasonxu1225/Uncertainty-aware-Inverse-Constrained-Reinforcement-Learning/tree/main
