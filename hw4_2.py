import random
import matplotlib.pyplot as plt
import gym
import mdptoolbox.mdp
from gym import envs, spaces
import pandas as pd
import stable_baselines3
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import time
import hiive.mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example
import mdptoolbox, mdptoolbox.example
import numpy as np


#### Forest Management
gamma = 0.9
P, R = hiive.mdptoolbox.example.forest() #default state number is 3, probability is 0.1
st = time.time()

def policy_iter_fm(gamma, P, R):
    iter = []
    time = []
    pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
    pi.run()
    policy = pi.policy
    iter.append(pi.iter)
    time.append(pi.time * 1000)
    return policy, time, iter


policy_fm_pi, time_fm_pi, iter_fm_pi = policy_iter_fm(gamma=gamma, P=P, R=R)


def value_iter_fm(gamma, P, R):
    iter = []
    time = []
    pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
    pi.run()
    policy = pi.policy
    iter.append(pi.iter)
    time.append(pi.time * 1000)
    return policy, time, iter


policy_fm, time_fm, iter_fm = value_iter_fm(gamma=gamma, P=P, R=R)


def Qlearning_fm(P, R,gamma):
    st = time.time()
    fm_ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=0.1, epsilon_decay=0.95, n_iter=1000000, alpha=0.95,
                                              skip_check=True)
    fm_ql.run()
    end = time.time()
    time_pass = end - st
    return time_pass, fm_ql.policy


time_pass, ql_policy_fm = Qlearning_fm(P, R, gamma)

def policy_reward_fm(state_range,model, count, gamma,eps):
    time_list = []
    iter_list = []
    reward_list = []
    for n_states in state_range:
        tot_episode = n_states*count
        P, R = hiive.mdptoolbox.example.forest(S=n_states)
        if model =='value':
            pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        elif model=='policy':
            pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
        elif model=='qlearner':
            pi = hiive.mdptoolbox.mdp.QLearning(P, R, gamma, epsilon=eps, epsilon_decay=0.95, n_iter=1000000, alpha=0.95,
                                              skip_check=True)
        pi.run()
        policy = pi.policy
        time_list.append(pi.time)
        if model != 'qlearner':
            iter_list.append(pi.iter)
        tot_reward = 0
        for s in range(n_states):
            state_reward = 0
            for state_episode in range(count):
                episode_reward = 0
                disc_r = 1
                while True:
                    a = policy[s]
                    prob = P[a][s]
                    possi_act = list(range(len(P[a][s])))
                    next_state = np.random.choice(possi_act, 1, p=prob)[0]
                    reward = R[s][a] * disc_r
                    episode_reward += reward
                    disc_r *= gamma
                    if next_state == 0:
                        break
                state_reward += episode_reward
            tot_reward += state_reward
        reward_list.append(tot_reward / count)
    return time_list, iter_list,reward_list

state_range = [3,100,200,300,400]
state_range = range(2,22,2)
time_fm_vi, iter_fm_vi, reward_fm_vi = policy_reward_fm(state_range=state_range,model='value',count=200,gamma=gamma,eps=None)
time_fm_pi, iter_fm_pi, reward_fm_pi = policy_reward_fm(state_range=state_range,model='policy',count=200,gamma=gamma,eps=None)

#Qlearner
time_fm_q, iter_fm_q, reward_fm_q = policy_reward_fm(state_range=state_range,model='qlearner',count=200,gamma=gamma,eps=0.1)
time_fm_q_eps, iter_fm_q_eps, reward_fm_q_eps = policy_reward_fm(state_range=state_range,model='qlearner',count=200,gamma=gamma,eps=0.9)

# reward
plt.plot(state_range,reward_fm_vi,label='Value Iteration')
plt.plot(state_range,reward_fm_pi,label='Policy Iteration')
plt.title('Compare Average Rewards by Different Number of States and Algorithms')
plt.xlabel('Number of States')
plt.ylabel('Average Rewards')
plt.legend()

plt.plot(state_range,iter_fm_vi,label='Value Iteration')
plt.plot(state_range,iter_fm_pi,label='Policy Iteration')
plt.title('Compare Iteration Numbers by Different Number of States and Algorithms')
plt.xlabel('Number of States')
plt.ylabel('Number of Iterations Needed')
plt.legend()

plt.plot(state_range,time_fm_vi,label='Value Iteration')
plt.plot(state_range,time_fm_pi,label='Policy Iteration')
plt.title('Compare Iteration Numbers by Different Number of States and Algorithms')
plt.xlabel('Number of States')
plt.ylabel('Run Time')
plt.legend()

# q learner reward
plt.plot(state_range,reward_fm_q_eps,label='Epsilon = 0.9')
plt.plot(state_range,reward_fm_q,label='Epsilon = 0.1')
plt.title('Compare Average Rewards by Different Number of States and Epsilon')
plt.xlabel('Number of States')
plt.ylabel('Average Rewards')
plt.legend()

plt.plot(state_range,time_fm_q_eps,label='Epsilon = 0.9')
plt.plot(state_range,time_fm_q,label='Epsilon = 0.1')
plt.title('Compare Average Rewards by Different Number of States and Epsilon')
plt.xlabel('Number of States')
plt.ylabel('Run Time')
plt.legend()

