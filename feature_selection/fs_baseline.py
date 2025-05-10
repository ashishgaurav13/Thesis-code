import basic
import utils
import random
import torch
import tqdm
import numpy as np
from scipy import stats
from skfeature.function.information_theoretical_based import CIFE, DISR, CMIM, MRMR
import atexit

atexit.register(utils.end_logging)
utils.start_logging("%s/0_log_baseline_%s.txt" % (basic.save_dir, basic.args.baseline))

if basic.env_name == 'highd':
    print("Using saved expert data for highd")
    expert_data = utils.generate_data(f'data/expert_data_{basic.env_name}.pt', c=basic.true_constraint_function, only_success=True)
else:
    expert_data = utils.generate_data(f'data/expert_data_{basic.env_name}_{basic.seed}.pt', c=basic.true_constraint_function, only_success=True)
nominal_data = utils.generate_data(f'data/zero_data_{basic.env_name}_{basic.seed}.pt', c=basic.zero_constraint_function)

data = []
for S, A in expert_data:
    for s, a in zip(S, A):
        data += [[basic.sa_func(s, a), 0]]
for S, A in nominal_data:
    for s, a in zip(S, A):
        data += [[basic.sa_func(s, a), 1]]
random.shuffle(data)

X = np.array([item[:-1] for item in data]).squeeze()
y = np.array([item[-1] for item in data])

if basic.args.baseline == 'cife':
    print("Order of features", CIFE.cife(X, y, n_selected_features=basic.n_features)[0])
elif basic.args.baseline == 'disr':
    print("Order of features", DISR.disr(X, y, n_selected_features=basic.n_features)[0])
elif basic.args.baseline == 'cmim':
    print("Order of features", CMIM.cmim(X, y, n_selected_features=basic.n_features)[0])
elif basic.args.baseline == 'mrmr':
    print("Order of features", MRMR.mrmr(X, y, n_selected_features=basic.n_features)[0])
