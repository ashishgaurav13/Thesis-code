import basic
import utils
import random
import torch
import tqdm
import numpy as np
from scipy import stats


def get_feature_i(i):
    data1 = []
    data2 = []
    for S, A in expert_data:
        for s, a in zip(S, A):
            data1 += [basic.sa_func(s, a)[i]]
    for S, A in nominal_data:
        for s, a in zip(S, A):
            data2 += [basic.sa_func(s, a)[i]]
    return data1, data2


expert_data = utils.generate_data(f'data/expert_data_{basic.env_name}.pt', c=basic.true_constraint_function, only_success=True)
nominal_data = utils.generate_data(f'data/zero_data_{basic.env_name}.pt', c=basic.zero_constraint_function)

items = []
for i in tqdm.trange(basic.n_features):
    if i in basic.skip:
        continue
    st = stats.stats.ks_2samp(*get_feature_i(i))
    items += [(i, st.statistic, st.pvalue)]


for i, statistic, pvalue in sorted(items, key=lambda x: (x[-1], -x[-2])):
    print("outer iteration", i, "KS statistic", statistic, "KS pvalue", pvalue)