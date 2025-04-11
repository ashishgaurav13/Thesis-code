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


def combine_dataset(data, label):
    data2 = []
    for S, A in data:
        for s, a in zip(S, A):
            data2 += [[basic.sa_func(s, a), label]]
    return data2


def get_combined_data(expert_data, nominal_data, convert_e=True, convert_n=True):
    data = []
    if convert_e:
        data += combine_dataset(expert_data, 0)
    else:
        data += expert_data
    if convert_n:
        data += combine_dataset(nominal_data, 1)
    else:
        data += nominal_data
    random.shuffle(data)
    return data


def f(itr, data, iii, cs=[], budget=1):
    D = [data]
    # create datasets based on conditioning
    for csi in cs[:budget]:
        values = [sa[csi] for sa, _ in data]
        if basic.continuous_f[csi]: # bucketize
            values = set(values)
            hist, bin_edges = np.histogram(list(values), bins=5)
            ranges = [(bin_edges[idx], bin_edges[idx+1]) for idx in range(len(bin_edges)-1)]
        else:
            values = set(values)
            ranges = list(values)
        DD = []
        for d in D:
            for k in ranges:
                if basic.continuous_f[csi]:
                    lo, hi = k
                    dd = [(sa, label) for sa, label in d if lo <= sa[csi] <= hi]
                else:
                    dd = [(sa, label) for sa, label in d if sa[csi] == k]
                DD += [dd]
        D = DD
    mis = []
    for di, d in tqdm.tqdm(enumerate(D)):
        d1 = np.array([sa[iii] for sa, label in d if label == 1])
        d2 = np.array([sa[iii] for sa, label in d if label == -1])
        d2t = torch.tensor(d2).float().view(-1, 1)
        d1t = torch.tensor(d1).float().view(-1, 1)
        if len(d1) > 0 and len(d2) > 0:
            ks2s = stats.stats.ks_2samp(d1, d2)
            mi, mi2 = ks2s.statistic, ks2s.pvalue
        else:
            mi, mi2 = 0, 1
        if np.isnan(mi):
            mis += [(mi, mi2)]
        else:
            mis += [(mi, mi2)]
    return sorted(mis, key=lambda x: x[0])[::-1][0]


expert_data = utils.generate_data(f'data/expert_data_{basic.env_name}.pt', c=basic.true_constraint_function, only_success=True)
nominal_data = utils.generate_data(f'data/zero_data_{basic.env_name}.pt', c=basic.zero_constraint_function)

itr = 1
basic.cs = []
selected = []
while True:
    if len(basic.cs) == 0:
        nominal_data = utils.generate_data(f'data/zero_data_{basic.env_name}.pt', c=basic.zero_constraint_function)
        combined_data = get_combined_data(expert_data, nominal_data, convert_e=True, convert_n=True)
    else:
        basic.input_dim = itr-1
        c, nominal_data = utils.icl(expert_data, N=basic.N, save_dir=basic.save_dir, outer_itr=itr-1)
        combined_data = get_combined_data(expert_data, nominal_data, convert_e=True, convert_n=True)
        torch.save([c.state_dict(), nominal_data], f"{basic.save_dir}/{itr-1}.pt")
    nad_value = utils.nad(expert_data, nominal_data)
    feature_n = len(combined_data[0][0])
    budget = feature_n
    mis = []
    for i in range(feature_n):
        if i in basic.cs or i in basic.skip:
            continue
        mi, mi2 = f(itr, combined_data, i, cs=basic.cs, budget=budget)
        mis += [(i, mi, mi2)]
    mis = sorted(mis, key=lambda x: x[2])
    n_z = 0
    for item in mis:
        if item[2] == 0.:
            n_z += 1
    if n_z > 1:
        mis = sorted(mis[:n_z], key=lambda x: x[1])[::-1]
    best_i, best_statistic, best_pvalue = mis[0]
    print(mis)
    print("Selected feature:", best_i, "Stat:", best_statistic, "pvalue:", best_pvalue, "nad_value:", nad_value)
    basic.cs += [best_i]
    selected += [(best_i, best_statistic, best_pvalue, nad_value)]
    itr += 1
    if len(basic.cs) >= feature_n-len(basic.skip):
        break

basic.input_dim = itr-1
c, nominal_data = utils.icl(expert_data, N=basic.N, save_dir=basic.save_dir, outer_itr=itr-1)
nad_value2 = utils.nad(expert_data, nominal_data)
torch.save([c.state_dict(), nominal_data], f"{basic.save_dir}/{itr-1}.pt")

for i, statistic, pvalue, nad in selected:
    print("outer iteration", i, "KS statistic", statistic, "KS pvalue", pvalue, "NAD before iteration", nad)
print("NAD post iteration", nad_value2)
