import itertools
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stable_baselines3.common import callbacks
from utils.data_utils import del_and_make
from utils.plot_utils import plot_curve
import ot
from scipy.spatial.distance import cdist

def beta_parameters_visualization(obs_str, constraint_net, alpha_all, beta_all, save_path):
    obs = np.asarray([[int(obs_str.split('-')[0]), int(obs_str.split('-')[1])]])
    tmp_data = constraint_net.prepare_data(obs=obs, acs=np.asarray([0]))
    alpha_beta = constraint_net.network(tmp_data)
    alpha = alpha_beta[:, 0].item()
    beta = alpha_beta[:, 1].item()
    print("alpha: {0}, beta: {1}".format(alpha, beta))
    alpha_all[obs_str].append(alpha)
    beta_all[obs_str].append(beta)
    plt.figure()
    plt.plot(range(len(alpha_all[obs_str])), alpha_all[obs_str])
    plt.savefig(save_path + '/alpha_{0}.png'.format(obs_str))
    plt.figure()
    plt.plot(range(len(beta_all[obs_str])), beta_all[obs_str])
    plt.savefig(save_path + '/beta_{0}.png'.format(obs_str))


def traj_visualization_2d(config, observations, save_path, model_name='', title='', axis_size=24):
    traj_num = len(observations)
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    plt.figure(figsize=(5, 5))
    for i in range(traj_num)[0: 5]:
        x = observations[i][:, config['env']["record_info_input_dims"][0]]
        y = observations[i][:, config['env']["record_info_input_dims"][1]]
        plt.plot(x, y, label='{0}th Traj'.format(i))
        plt.scatter(x, y)
    xticks = np.arange(config['env']["visualize_info_ranges"][0][0],
                       config['env']["visualize_info_ranges"][0][1] + 1, 1)
    plt.xticks(xticks)
    yticks = np.arange(config['env']["visualize_info_ranges"][1][0],
                       config['env']["visualize_info_ranges"][1][1] + 1, 1)
    plt.yticks(yticks)
    # plt.yticks(config['env']["visualize_info_ranges"][1])
    # plt.xlabel(config['env']["record_info_names"][0], fontsize=axis_size)
    # plt.ylabel(config['env']["record_info_names"][1], fontsize=axis_size)
    plt.legend(fontsize=15, loc='lower right')
    plt.grid(linestyle='--')
    plt.title('{0}'.format(title), fontsize=axis_size)
    plt.savefig(os.path.join(save_path, "2d_traj_visual_{0}_{1}.png".format(model_name, title)))

def wasserstein_distance2d(u, v, p='cityblock'):
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape and len(u.shape) == 2)
    dim1, dim2 = u.shape
    assert(p in ['euclidean', 'cityblock'])
    coords = np.zeros((dim1*dim2, 2)).astype('float')
    for i in range(dim1):
        for j in range(dim2):
            coords[i*dim2+j, :] = [i, j]
    d = cdist(coords, coords, p)
    u /= u.sum()
    v /= v.sum()
    return ot.emd2(u.flatten(), v.flatten(), d)

def gridworld_imshow(m, fig, ax):
    m = np.array(m).squeeze()
    assert len(m.shape) == 2
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    m = (m-np.min(m))/(np.max(m)-np.min(m)+1e-3)
    im = ax.imshow(m, cmap="gray")
    # im.set_clim(0, 1)
    ax.set_xticks(np.arange(m.shape[0]))
    ax.set_yticks(np.arange(m.shape[1]))
    cbar = fig.colorbar(im, cax=cax)

def discount(x, config, invert=False):
    n = len(x)
    g = 0
    d = []
    for i in range(n):
        if not invert:
            g = x[n - 1 - i] + config['env']['cost_gamma'] * g
            d = [g] + d
        else:
            g = g + x[i] * (config['env']['cost_gamma']**i)
            d = d + [g]
    return d

def psat(env_id, config, obs_dim, acs_dim, observations, actions, cost_function, plot=False, save_to=None):
    traj_num = len(observations)
    import matplotlib as mpl
    budget = config['PPO']['budget']

    if env_id == 'CustomCartPole':
        inds = []
        Gc0s = []
        for i in range(traj_num):
            S = observations[i][:, 0]
            n = len(S)
            input_all = np.zeros((n, obs_dim + acs_dim))
            input_all[:, 0] = S
            input_all[:, obs_dim] = 0
            obs = input_all[:, :obs_dim]
            acs = input_all[:, obs_dim:]
            preds = cost_function(obs=obs, acs=acs)
            dis_preds = discount(preds, config)
            Gc0 = dis_preds[0]
            Gc0s += [Gc0]
            inds += [int(Gc0 <= budget)]
        Gc0s = np.array(Gc0s)
        print("PSAT:", sum(inds)/traj_num)
        if plot:
            import seaborn as sns
            sns.set_style('darkgrid')
            sns.set(rc={
                "text.usetex": True,
                "font.family": "cmr",
                "font.size": 14,
            })
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.hist(Gc0s, bins=20)
            print(list(Gc0s))
            ax.axvline(x=np.mean(Gc0s), color='red')
            ax.axvline(x=20, color='green')
            ax.set(ylabel='')
            ax.set_title('UAICRL')
            fig.tight_layout()
            fig.savefig(save_to)

    elif env_id == 'Gridworld':
        inds = []
        Gc0s = []
        for i in range(traj_num):
            X = observations[i][:, 0]
            Y = observations[i][:, 1]
            A = actions[i]
            n = len(A)
            input_all = np.zeros((n, obs_dim + acs_dim))
            input_all[:, 0] = X
            input_all[:, 1] = Y
            input_all[:, obs_dim] = A
            obs = input_all[:, :obs_dim]
            acs = input_all[:, obs_dim:]
            preds = cost_function(obs=obs, acs=acs)
            dis_preds = discount(preds, config)
            Gc0 = dis_preds[0]
            Gc0s += [Gc0]
            inds += [int(Gc0 <= budget)]
        Gc0s = np.array(Gc0s)
        print("PSAT:", sum(inds)/traj_num)
        if plot:
            import seaborn as sns
            sns.set_style('darkgrid')
            sns.set(rc={
                "text.usetex": True,
                "font.family": "cmr",
                "font.size": 14,
            })
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.hist(Gc0s, bins=20)
            print(list(Gc0s))
            ax.axvline(x=np.mean(Gc0s), color='red')
            ax.axvline(x=0.99, color='green')
            ax.set(ylabel='')
            ax.set_title('UAICRL')
            fig.tight_layout()
            fig.savefig(save_to)

    elif env_id in ['AntWall', 'HalfCheetahWithPos']:
        inds = []
        Gc0s = []
        for i in range(traj_num):
            S = observations[i][:, 0]
            n = len(S)
            input_all = np.zeros((n, obs_dim + acs_dim))
            input_all[:, 0] = S
            input_all[:, obs_dim] = 0
            obs = input_all[:, :obs_dim]
            acs = input_all[:, obs_dim:]
            preds = cost_function(obs=obs, acs=acs)
            dis_preds = discount(preds, config)
            Gc0 = dis_preds[0]
            Gc0s += [Gc0]
            inds += [int(Gc0 <= budget)]
        Gc0s = np.array(Gc0s)
        print("PSAT:", sum(inds)/traj_num)
        if plot:
            import seaborn as sns
            sns.set_style('darkgrid')
            sns.set(rc={
                "text.usetex": True,
                "font.family": "cmr",
                "font.size": 14,
            })
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.hist(Gc0s, bins=20)
            print(list(Gc0s))
            ax.axvline(x=np.mean(Gc0s), color='red')
            ax.axvline(x=15, color='green')
            ax.set(ylabel='')
            ax.set_title('UAICRL')
            fig.tight_layout()
            fig.savefig(save_to)

    if save_to != None:
        torch.save(list(Gc0s), save_to.rstrip(".png")+".pt")
    # exit(0)

def traj_visualization_2d_new(env_id, config, observations, actions, expert_obs, expert_acs, save_path, model_name='', title='', axis_size=24):
    traj_num = len(observations)
    import matplotlib as mpl
    
    if env_id == 'CustomCartPole':
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        accrual = np.zeros_like(np.arange(-2.4, 2.4+0.1, 0.1))
        for i in range(traj_num):
            S = observations[i][:, 0]
            for s in S:
                bin_nbr = np.clip(int(np.floor((s+2.4+0.1/2)/0.1)), 0, 49)
                accrual[bin_nbr] += 1
        accrual /= (accrual.max()+1e-6)
        ax[0].plot(np.arange(-2.4, 2.4+0.1, 0.1), accrual, color='green')
        ax[0].margins(0.05, 0.05)
        ax[0].set_title('{0}'.format(title)) #, fontsize=axis_size)
        accrual_expert = np.zeros_like(np.arange(-2.4, 2.4+0.1, 0.1))
        for i in range(expert_obs.shape[0]):
            s = expert_obs[i, 0]
            bin_nbr = np.clip(int(np.floor((s+2.4+0.1/2)/0.1)), 0, 49)
            accrual_expert[bin_nbr] += 1
        accrual_expert /= (accrual_expert.max()+1e-6)
        ax[1].plot(np.arange(-2.4, 2.4+0.1, 0.1), accrual_expert, color='green')
        ax[1].margins(0.05, 0.05)
        ax[1].set_title('{0} (expert)'.format(title)) #, fontsize=axis_size)
        fig.tight_layout()

        rng = np.arange(-2.4, 2.4+0.1, 0.1)
        accrual = np.zeros((2, len(rng)))
        for i in range(traj_num):
            S = observations[i][:, 0]
            A = actions[i]
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s+2.4+0.1/2)/0.1)), 0, 49)
                accrual[int(a)][bin_nbr] += 1
        accrual[0, :] /= (accrual[0, :].max()+1e-6)
        accrual[1, :] /= (accrual[1, :].max()+1e-6)
        expert_accrual = np.zeros((2, len(rng)))
        for i in range(expert_obs.shape[0]):
            s = expert_obs[i, 0]
            a = expert_acs[i]
            bin_nbr = np.clip(int(np.floor((s+2.4+0.1/2)/0.1)), 0, 49)
            expert_accrual[int(a)][bin_nbr] += 1
        expert_accrual[0, :] /= (expert_accrual[0, :].max()+1e-6)
        expert_accrual[1, :] /= (expert_accrual[1, :].max()+1e-6)
        accrual_expert = expert_accrual
        nad = 0.5*(wasserstein_distance2d(expert_accrual[0, :].reshape(1, -1), accrual[0, :].reshape(1, -1))+\
            wasserstein_distance2d(expert_accrual[1, :].reshape(1, -1), accrual[1, :].reshape(1, -1)))
        print("NAD: %.2f" % nad)
    
    elif env_id == 'Gridworld':
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        accrual = np.array([np.zeros((7, 7)) for _ in range(8)])
        for i in range(traj_num):
            X = observations[i][:, 0]
            Y = observations[i][:, 1]
            A = actions[i]
            for x,y,a in zip(X,Y,A):
                accrual[a][x][y] += 1
        accrual = np.mean(accrual, axis=0)
        accrual /= (np.max(accrual)+1e-6)
        gridworld_imshow(accrual, fig, ax[0])
        ax[0].set_title('{0}'.format(title)) #, fontsize=axis_size)
        accrual_expert = np.array([np.zeros((7, 7)) for _ in range(8)])
        for i in range(expert_obs.shape[0]):
            x = expert_obs[i, 0]
            y = expert_obs[i, 1]
            a = expert_acs[i]
            accrual_expert[int(a)][x][y] += 1
        accrual_expert = np.mean(accrual_expert, axis=0)
        accrual_expert /= (np.max(accrual_expert)+1e-6)
        gridworld_imshow(accrual_expert, fig, ax[1])
        ax[1].set_title('{0} (expert)'.format(title)) #, fontsize=axis_size)
        fig.tight_layout()
        nad = wasserstein_distance2d(accrual_expert, accrual)
        print("NAD: %.2f" % nad)

    elif env_id in ['AntWall', 'HalfCheetahWithPos']:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        accrual = np.zeros_like(np.arange(-5, 5+0.1, 0.1))
        for i in range(traj_num):
            S = observations[i][:, 0]
            for s in S:        
                if -5 <= s <= 5:
                    bin_nbr = np.clip(int(np.floor((s+5+0.1/2)/0.1)), 0, 100)
                    accrual[bin_nbr] += 1
        accrual /= (accrual.max()+1e-6)
        ax[0].plot(np.arange(-5, 5+0.1, 0.1), accrual, color='green')
        ax[0].margins(0.05, 0.05)
        ax[0].set_title('{0}'.format(title)) #, fontsize=axis_size)
        accrual_expert = np.zeros_like(np.arange(-5, 5+0.1, 0.1))
        for i in range(expert_obs.shape[0]):
            s = expert_obs[i, 0]
            if -5 <= s <= 5:
                bin_nbr = np.clip(int(np.floor((s+5+0.1/2)/0.1)), 0, 100)
                accrual_expert[bin_nbr] += 1
        accrual_expert /= (accrual_expert.max()+1e-6)
        ax[1].plot(np.arange(-5, 5+0.1, 0.1), accrual_expert, color='green')
        ax[1].margins(0.05, 0.05)
        ax[1].set_title('{0} (expert)'.format(title)) #, fontsize=axis_size)
        fig.tight_layout()

        nad = wasserstein_distance2d(accrual_expert.reshape(1, -1), accrual.reshape(1, -1))
        print("NAD: %.2f" % nad)


    torch.save([accrual_expert, accrual], os.path.join(save_path, "2d_traj_visual_{0}_{1}.pt".format(model_name, title)))
    fig.savefig(os.path.join(save_path, "2d_traj_visual_{0}_{1}.png".format(model_name, title)))

def traj_visualization_1d(config, observations, save_path):
    for record_info_idx in range(len(config['env']["record_info_names"])):
        plt.figure()
        record_info_name = config['env']["record_info_names"][record_info_idx]
        record_obs_dim = config['env']["record_info_input_dims"][record_info_idx]
        if config['running']['store_by_game']:
            plt.hist(np.concatenate(observations, axis=0)[:, record_obs_dim],
                     bins=40, )
        else:
            plt.hist(observations[:, record_info_idx],
                     bins=40, )
        plt.legend()
        plt.savefig(os.path.join(save_path, "{0}_traj_visual.png".format(record_info_name)))


def constraint_visualization_1d(cost_function, feature_range, select_dim, obs_dim, acs_dim,
                                save_name, device='cpu', feature_data=None, feature_cost=None, feature_name=None,
                                empirical_input_means=None, num_points=1000, axis_size=24):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    selected_feature_generation = np.linspace(feature_range[0], feature_range[1], num_points)
    if empirical_input_means is None:
        input_all = np.zeros((num_points, obs_dim + acs_dim))
    else:
        assert len(empirical_input_means) == obs_dim + acs_dim
        input_all = np.expand_dims(empirical_input_means, 0).repeat(num_points, axis=0)
        # input_all = torch.tensor(input_all)
    input_all[:, select_dim] = selected_feature_generation
    with torch.no_grad():
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        preds = cost_function(obs=obs, acs=acs, force_mode='mean')  # use the mean of a distribution for visualization
    ax[0].plot(selected_feature_generation, preds, c='r', linewidth=5)
    if feature_data is not None:
        ax[0].scatter(feature_data, feature_cost)
        ax[1].hist(feature_data, bins=40, range=(feature_range[0], feature_range[1]))
        ax[1].set_axisbelow(True)
        # Turn on the minor TICKS, which are required for the minor GRID
        ax[1].minorticks_on()
        ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax[1].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax[1].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
        ax[1].set_ylabel('Frequency', fontdict={'fontsize': axis_size})
    ax[0].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
    ax[0].set_ylabel('Cost', fontdict={'fontsize': axis_size})
    ax[0].set_ylim([0, 1])
    ax[0].set_xlim(feature_range)
    ax[0].set_axisbelow(True)
    # Turn on the minor TICKS, which are required for the minor GRID
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax[0].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    fig.savefig(save_name)
    plt.close(fig=fig)


def constraint_visualization_2d(cost_function, feature_range, select_dims,
                                obs_dim, acs_dim,
                                title='', num_points_per_feature=100,
                                axis_size=20, force_mode=None, save_path=None, empirical_input_means=None, model_name=''):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size

    selected_feature_1_generation = np.linspace(feature_range[0][0], feature_range[0][1], num_points_per_feature)
    selected_feature_2_generation = np.linspace(feature_range[1][1], feature_range[1][0], num_points_per_feature)
    selected_feature_all = np.asarray(
        [d for d in itertools.product(selected_feature_1_generation, selected_feature_2_generation)])
    tmp = selected_feature_all.reshape([num_points_per_feature, num_points_per_feature, 2]).transpose(1, 0, 2)
    if empirical_input_means is None:
        input_all = np.zeros((num_points_per_feature ** 2, obs_dim + acs_dim))
    else:
        assert len(empirical_input_means) == obs_dim + acs_dim
        input_all = np.expand_dims(empirical_input_means, 0).repeat(num_points_per_feature ** 2, axis=0)
    input_all[:, select_dims[0]] = selected_feature_all[:, 0]
    input_all[:, select_dims[1]] = selected_feature_all[:, 1]

    obs = input_all[:, :obs_dim]
    acs = input_all[:, obs_dim:]

    with torch.no_grad():
        if force_mode is None:
            preds = cost_function(obs=obs, acs=acs)
        else:
            preds = cost_function(obs=obs, acs=acs, force_mode=force_mode)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(preds.reshape([num_points_per_feature, num_points_per_feature]).transpose(1, 0),
                   cmap='binary',  # 'cool',
                   interpolation="nearest",
                   extent=[feature_range[0][0], feature_range[0][1],
                           feature_range[1][0], feature_range[1][1]])
    cbar = plt.colorbar(im)
    cbar.set_label("Cost", fontsize=axis_size)
    xticks = np.arange(feature_range[0][0],
                       feature_range[0][1] + 1, 1)
    plt.xticks(xticks)
    yticks = np.arange(feature_range[1][0],
                       feature_range[1][1] + 1, 1)
    plt.yticks(yticks)
    plt.grid(linestyle='--', color='r', alpha=1)
    plt.title('{0}'.format(title), fontsize=axis_size)
    # plt.show()
    plt.savefig(os.path.join(save_path, "constraint_visualization_{0}_{1}.png".format(model_name,title)))

def mse(u, v):
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape)
    return np.mean(np.power(u-v, 2))

def constraint_visualization_2d_new(env_id, cost_function, obs_dim, acs_dim,
                                title='', force_mode=None, save_path=None, model_name='',
                                gt_cost_function=None):
    import matplotlib as mpl

    if env_id == 'CustomCartPole':
        if gt_cost_function != None:
            fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax = [ax]
        selected_feature_generation = np.arange(-2.4, 2.4+0.1, 0.1)
        n = len(selected_feature_generation)
        input_all = np.zeros((n, obs_dim + acs_dim))
        input_all[:, 0] = selected_feature_generation
        input_all[:, obs_dim] = 0
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        if force_mode is None:
            preds_0 = cost_function(obs=obs, acs=acs)
        else:
            preds_0 = cost_function(obs=obs, acs=acs, force_mode=force_mode)
        input_all[:, obs_dim] = 1
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        if force_mode is None:
            preds_1 = cost_function(obs=obs, acs=acs)
        else:
            preds_1 = cost_function(obs=obs, acs=acs, force_mode=force_mode)
        preds = np.array([preds_0, preds_1])
        ax[0].plot(np.arange(-2.4, 2.4+0.1, 0.1), preds_0, label="a=0", color='blue')
        ax[0].plot(np.arange(-2.4, 2.4+0.1, 0.1), preds_1, label="a=1", color='red')
        ax[0].legend(loc='best')
        ax[0].set_ylim(0-0.05, 1+0.05)
        ax[0].margins(0.05, 0.05)
        ax[0].set_title('{0}'.format(title)) #, fontsize=axis_size)
        if gt_cost_function != None:
            selected_feature_generation = np.arange(-2.4, 2.4+0.1, 0.1)
            n = len(selected_feature_generation)
            input_all = np.zeros((n, obs_dim + acs_dim))
            input_all[:, 0] = selected_feature_generation
            input_all[:, obs_dim] = 0
            obs = input_all[:, :obs_dim]
            acs = input_all[:, obs_dim:]
            preds_0 = gt_cost_function(obs=obs, acs=acs)
            input_all[:, obs_dim] = 1
            obs = input_all[:, :obs_dim]
            acs = input_all[:, obs_dim:]
            preds_1 = gt_cost_function(obs=obs, acs=acs)
            gt_preds = np.array([preds_0, preds_1])
            ax[1].plot(np.arange(-2.4, 2.4+0.1, 0.1), preds_0, label="a=0", color='blue')
            ax[1].plot(np.arange(-2.4, 2.4+0.1, 0.1), preds_1, label="a=1", color='red')
            ax[1].legend(loc='best')
            ax[1].set_ylim(0-0.05, 1+0.05)
            ax[1].margins(0.05, 0.05)
            ax[1].set_title('{0} (GT)'.format(title)) #, fontsize=axis_size)

            cmse = mse(gt_preds, preds)
            print("CMSE: %.2f" % cmse)

        fig.tight_layout()
    
    elif env_id == 'Gridworld':
        if gt_cost_function != None:
            fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax = [ax]
        selected_feature_generation = np.arange(0, 7, 1)
        l = np.array([[i, j] for i in selected_feature_generation for j in selected_feature_generation])
        input_all = np.zeros((len(l), obs_dim + acs_dim))
        input_all[:, 0] = l[:, 0]
        input_all[:, 1] = l[:, 1]
        obs = input_all[:, :obs_dim]
        preds_all = []
        for a in range(8):
            input_all[:, obs_dim] = a
            acs = input_all[:, obs_dim:]
            if force_mode is None:
                preds = cost_function(obs=obs, acs=acs)
            else:
                preds = cost_function(obs=obs, acs=acs, force_mode=force_mode)
            preds_all += [preds]
        preds_all = np.array(preds_all)
        preds = np.mean(preds_all, axis=0)
        preds = np.reshape(preds, (7, 7))
        gridworld_imshow(preds, fig, ax[0])
        ax[0].set_title('{0}'.format(title)) #, fontsize=axis_size)
        if gt_cost_function != None:
            input_all = np.zeros((len(l), obs_dim + acs_dim))
            input_all[:, 0] = l[:, 0]
            input_all[:, 1] = l[:, 1]
            obs = input_all[:, :obs_dim]
            preds_all2 = []
            for a in range(8):
                input_all[:, obs_dim] = a
                acs = input_all[:, obs_dim:]
                preds2 = gt_cost_function(obs=obs, acs=acs)
                preds_all2 += [preds2]
            preds_all2 = np.array(preds_all2)
            preds_expert = np.mean(preds_all2, axis=0)
            preds_expert = np.reshape(preds_expert, (7, 7))
            gridworld_imshow(preds_expert, fig, ax[1])
            ax[1].set_title('{0} (GT)'.format(title)) #, fontsize=axis_size)
            cmse = mse(preds_expert, preds)
            print("CMSE: %.2f" % cmse)

        fig.tight_layout()

    elif env_id == 'AntWall' or env_id == 'HalfCheetahWithPos':

        if gt_cost_function != None:
            fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax = [ax]
        selected_feature_generation = np.arange(-5, 5+0.1, 0.1)
        n = len(selected_feature_generation)
        input_all = np.zeros((n, obs_dim + acs_dim))
        input_all[:, 0] = selected_feature_generation
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        if force_mode is None:
            preds = cost_function(obs=obs, acs=acs)
        else:
            preds = cost_function(obs=obs, acs=acs, force_mode=force_mode)
        ax[0].plot(np.arange(-5, 5+0.1, 0.1), preds, color='pink')
        ax[0].set_ylim(0-0.05, 1+0.05)
        ax[0].margins(0.05, 0.05)
        ax[0].set_title('{0}'.format(title)) #, fontsize=axis_size)
        if gt_cost_function != None:
            selected_feature_generation = np.arange(-5, 5+0.1, 0.1)
            n = len(selected_feature_generation)
            input_all = np.zeros((n, obs_dim + acs_dim))
            input_all[:, 0] = selected_feature_generation
            obs = input_all[:, :obs_dim]
            acs = input_all[:, obs_dim:]
            gt_preds = gt_cost_function(obs=obs, acs=acs)
            ax[1].plot(np.arange(-5, 5+0.1, 0.1), gt_preds, color='pink')
            ax[1].set_ylim(0-0.05, 1+0.05)
            ax[1].margins(0.05, 0.05)
            ax[1].set_title('{0} (GT)'.format(title)) #, fontsize=axis_size)
            cmse = mse(gt_preds, preds)
            print("CMSE: %.2f" % cmse)

        fig.tight_layout() 

    # plt.show()
    torch.save(preds, os.path.join(save_path, "constraint_visualization_{0}_{1}.pt".format(model_name,title)))
    fig.savefig(os.path.join(save_path, "constraint_visualization_{0}_{1}.png".format(model_name,title)))


class PlotCallback(callbacks.BaseCallback):
    """
    This callback can be used/modified to fetch something from the buffer and make a
    plot using some custom plot function.
    """

    def __init__(
            self,
            train_env_id: str,
            plot_freq: int = 10000,
            log_path: str = None,
            plot_save_dir: str = None,
            verbose: int = 1,
            name_prefix: str = "model",
            plot_feature_names_dims: dict = {},
    ):
        super(PlotCallback, self).__init__(verbose)
        self.name_prefix = name_prefix
        self.log_path = log_path
        self.plot_save_dir = plot_save_dir
        self.plot_feature_names_dims = plot_feature_names_dims

    def _init_callback(self):
        # Make directory to save plots
        # del_and_make(os.path.join(self.plot_save_dir, "plots"))
        # self.plot_save_dir = os.path.join(self.plot_save_dir, "plots")
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        try:
            obs = self.model.rollout_buffer.orig_observations.copy()
        except:  # PPO uses rollout buffer which does not store orig_observations
            obs = self.model.rollout_buffer.observations.copy()
            # unormalize observations
            obs = self.training_env.unnormalize_obs(obs)
        obs = obs.reshape(-1, obs.shape[-1])  # flatten the batch size and num_envs dimensions
        rewards = self.model.rollout_buffer.rewards.copy()
        for record_info_name in self.plot_feature_names_dims.keys():
            plot_record_infos, plot_costs = zip(
                *sorted(zip(obs[:, self.plot_feature_names_dims[record_info_name]], rewards)))
            path = os.path.join(self.plot_save_dir, f"{self.name_prefix}_{self.num_timesteps}_steps")
            if not os.path.exists(path):
                os.mkdir(path)
            plot_curve(draw_keys=[record_info_name],
                       x_dict={record_info_name: plot_record_infos},
                       y_dict={record_info_name: plot_costs},
                       xlabel=record_info_name,
                       ylabel='cost',
                       save_name=os.path.join(path, "{0}_visual.png".format(record_info_name)),
                       apply_scatter=True
                       )
        # self.plot_fn(obs, os.path.join(self.plot_save_dir, str(self.num_timesteps) + ".png"))

# class CNSEvalCallback(EventCallback):
#     """
#     Callback for evaluating an agent.
#
#     :param eval_env: The environment used for initialization
#     :param callback_on_new_best: Callback to trigger
#         when there is a new best model according to the ``mean_reward``
#     :param n_eval_episodes: The number of episodes to test the agent
#     :param eval_freq: Evaluate the agent every eval_freq call of the callback.
#     :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
#         will be saved. It will be updated at each evaluation.
#     :param best_model_save_path: Path to a folder where the best model
#         according to performance on the eval env will be saved.
#     :param deterministic: Whether the evaluation should
#         use a stochastic or deterministic actions.
#     :param deterministic: Whether to render or not the environment during evaluation
#     :param render: Whether to render or not the environment during evaluation
#     :param verbose:
#     """
#
#     def __init__(
#             self,
#             eval_env: Union[gym.Env, VecEnv],
#             callback_on_new_best: Optional[BaseCallback] = None,
#             n_eval_episodes: int = 5,
#             eval_freq: int = 10000,
#             log_path: str = None,
#             best_model_save_path: str = None,
#             deterministic: bool = True,
#             render: bool = False,
#             verbose: int = 1,
#             record_info_names: list = [],
#             callback_for_evaluate_policy: Optional[Callable] = None
#     ):
#         super(CNSEvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
#         self.n_eval_episodes = n_eval_episodes
#         self.record_info_names = record_info_names
#         self.plot_freq = eval_freq
#         self.best_mean_reward = -np.inf
#         self.last_mean_reward = -np.inf
#         self.deterministic = deterministic
#         self.render = render
#         self.callback_for_evaluate_policy = callback_for_evaluate_policy
#
#         # Convert to VecEnv for consistency
#         if not isinstance(eval_env, VecEnv):
#             eval_env = DummyVecEnv([lambda: eval_env])
#
#         if isinstance(eval_env, VecEnv):
#             assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"
#
#         self.eval_env = eval_env
#         self.best_model_save_path = best_model_save_path
#         # Logs will be written in ``evaluations.npz``
#         if log_path is not None:
#             log_path = os.path.join(log_path, "evaluations")
#         self.log_path = log_path
#         self.evaluations_results = []
#         self.evaluations_timesteps = []
#         self.evaluations_length = []
#
#     def _init_callback(self):
#         # Does not work in some corner cases, where the wrapper is not the same
#         if not isinstance(self.training_env, type(self.eval_env)):
#             warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")
#
#         # Create folders if needed
#         if self.best_model_save_path is not None:
#             os.makedirs(self.best_model_save_path, exist_ok=True)
#         if self.log_path is not None:
#             os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
#
#     def _on_step(self) -> bool:
#
#         if self.plot_freq > 0 and self.n_calls % self.plot_freq == 0:
#             # Sync training and eval env if there is VecNormalize
#             sync_envs_normalization(self.training_env, self.eval_env)
#
#             average_true_reward, std_true_reward, record_infos, costs = \
#                 evaluate_icrl_policy(self.model, self.eval_env,
#                                      record_info_names=self.record_info_names,
#                                      n_eval_episodes=self.n_eval_episodes,
#                                      deterministic=False)
#             for record_info_idx in range(len(self.record_info_names)):
#                 record_info_name = self.record_info_names[record_info_idx]
#                 plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name], costs)))
#                 if len(self.expert_acs.shape) == 1:
#                     empirical_input_means = np.concatenate([self.expert_obs, np.expand_dims(self.expert_acs, 1)], axis=1).mean(0)
#                 else:
#                     empirical_input_means = np.concatenate([self.expert_obs, self.expert_acs], axis=1).mean(0)
#                 plot_constraints(cost_function=constraint_net.cost_function,
#                                  feature_range=config['env']["visualize_info_ranges"][record_info_idx],
#                                  select_dim=config['env']["record_info_input_dims"][record_info_idx],
#                                  obs_dim=constraint_net.obs_dim,
#                                  acs_dim=1 if is_discrete else constraint_net.acs_dim,
#                                  device=constraint_net.device,
#                                  save_name=os.path.join(path, "{0}_visual.png".format(record_info_name)),
#                                  feature_data=plot_record_infos,
#                                  feature_cost=plot_costs,
#                                  feature_name=record_info_name,
#                                  empirical_input_means=empirical_input_means)
#
#             if self.log_path is not None:
#                 self.evaluations_timesteps.append(self.num_timesteps)
#                 self.evaluations_results.append(episode_rewards)
#                 self.evaluations_length.append(episode_lengths)
#                 np.savez(
#                     self.log_path,
#                     timesteps=self.evaluations_timesteps,
#                     results=self.evaluations_results,
#                     ep_lengths=self.evaluations_length,
#                 )
#
#             mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
#             mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
#             self.last_mean_reward = mean_reward
#
#             if self.verbose > 0:
#                 print(
#                     f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
#                 print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
#             # Add to current Logger
#             self.logger.record("eval/mean_reward", float(mean_reward))
#             self.logger.record("eval/mean_ep_length", mean_ep_length)
#             self.logger.record("eval/best_mean_reward", max(self.best_mean_reward, mean_reward))
#
#             if mean_reward > self.best_mean_reward:
#                 if self.verbose > 0:
#                     print("New best mean reward!")
#                 if self.best_model_save_path is not None:
#                     self.model.save(os.path.join(self.best_model_save_path, "best_model"))
#                 self.best_mean_reward = mean_reward
#                 # Trigger callback if needed
#                 if self.callback is not None:
#                     return self._on_event()
#
#         return True
#
#     def update_child_locals(self, locals_: Dict[str, Any]) -> None:
#         """
#         Update the references to the local variables.
#
#         :param locals_: the local variables during rollout collection
#         """
#         if self.callback:
#             self.callback.update_locals(locals_)
