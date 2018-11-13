import mdp
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import math



def normalize(vals):
    """
    normalize to (0, max_val)
    input:
    vals: 1d array
    """
    min_val = torch.min(vals)
    max_val = torch.max(vals)
    return (vals - min_val) / (max_val - min_val)


def state_visitation_frequencies(M,trajectories):
    freq = torch.zeros(M.n_states)
    for n in range(trajectories.shape[0]):
        for state in trajectories[n,:,0]:
            freq[state.long()] += 1

    assert(torch.sum(freq) == (trajectories.shape[0] * trajectories.shape[1]))

    freq /= trajectories.shape[0]

    return freq

def get_policy_w_r(M,reward_vector,values=None):
    if values is None:

        values = M.optimum_state_value(epsilon=0.001,reward_vector=reward_vector)

    policy = M.get_policy(values=values,deterministic=True)
    return policy, values

def find_expected_frequency(M, reward, trajectories):
    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]
    policy, val = get_policy_w_r(M,reward_vector=reward)
    N=trajectory_length


    starting_states_freq = torch.bincount(trajectories[:,0,0].long(), minlength=M.n_states)
    starting_states_freq = starting_states_freq.float()
    starting_states_prob = starting_states_freq / n_trajectories
    expected_freq = torch.zeros()

    svf = starting_states_prob.float()
    svf_ = torch.zeros(M.n_states, dtype=torch.float)
    total_svf = torch.zeros(M.n_states, dtype=torch.float)
    total_svf += svf
    for _ in range(1, N):
        for s_ in range(M.n_states):
            for s in range(M.n_states):
                svf_[s_] = (M.transition_probability[s,:,s_] @ policy[s,:]) * svf[s]

        svf = torch.tensor(svf_)
        total_svf += svf
    return total_svf, val, policy

def feature_expectations(M,trajectories):
    freq = state_visitation_frequencies(M,trajectories)
    feature_expectations = freq @ M.feature_matrix
    return feature_expectations

def softmax(x_1, x_2):
    max_x = max(x_1,x_2)
    min_x = min(x_1, x_2)
    return max_x + torch.log(1 + torch.exp(min_x - max_x))


def evd(M,reward,starting_probs,optimal_value_function, policy=None):
    if policy is None:
        subopt_value_func = M.optimum_state_value(reward_vector=reward)
        subopt_policy = M.get_policy(values=subopt_value_func, deterministic=True)

    else:
        subopt_policy = policy

    true_reward_subopt_value = M.state_value(policy=subopt_policy,reward_vector=None)


    if starting_probs is None:
        starting_probs = torch.ones(M.n_states, dtype=torch.float) / M.n_states

    return (optimal_value_function @ starting_probs) - (true_reward_subopt_value @ starting_probs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9,3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Standard_Maxent(nn.Module):
    def __init__(self):
        super(Standard_Maxent, self).__init__()
        self.fc1 = nn.Linear(9,1)


    def forward(self, x):
        x = self.fc1(x)
        return x

torch.manual_seed(100)
experiments = list()
rew_lin = torch.zeros(9)
rew_lin[0] = 0.9
rew_lin[1:] = 0.1
M = mdp.Binary_World(8, 0, 0.9, prob_blue=0.4)
optim_value_func = M.optimum_state_value()
deterministic_policy = M.get_policy(values=optim_value_func)
all_trajectories = M.generate_trajectories(n_trajectory=128, horizon=40, policy=deterministic_policy)

data = M.feature_matrix.unsqueeze(0)
data = torch.tensor(data,requires_grad=True)
print(M.reward_vector)

for mdp_num in range(10):
    print("Replication:{}".format(mdp_num))
    best_counter = 20
    for experiment in [128,64,32,16,8,4]:

        trajectories = all_trajectories[0:experiment,:,:]
        svf = state_visitation_frequencies(M,trajectories)

        net = Net()
        optimizer = torch.optim.Adam(net.parameters())
        for epoch in range(20):
            if best_counter == 0:
                break
            print("Experiment: {}".format(experiment))
            optimizer.zero_grad()
            net_out = net(data)
            evf, val_w_r, subopt_pol = find_expected_frequency(M,net_out.squeeze(0).squeeze(1),trajectories=trajectories)

            diff = svf - evf
            evd_current = evd(M,normalize(net_out.squeeze(1)),None,optim_value_func, policy=subopt_pol)
            if epoch ==0:
                best_evd = evd_current

            if(evd_current < best_evd):
                best_counter = 20
                best_evd = evd_current
            else:
                best_counter = best_counter - 1

            print(best_evd)
            print(evd_current)

            net_out.backward(diff.unsqueeze(1).unsqueeze(0))
            optimizer.step()

        experiments.append((mdp_num,experiment,best_evd,evd_current))
        print(net_out)
with open('experiments_deep_binaryworld_h5_rep10.pkl', 'wb') as f:
    pickle.dump(experiments, f)

