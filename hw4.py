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
import mdptoolbox, mdptoolbox.example
import numpy as np


np.random.seed(1105)
#### Frozen Lake - big
map = generate_random_map(size=20, p=0.995)
env = gym.make("FrozenLake-v1", is_slippery=True,desc=map)
env.render()

# value iteration
def value_iteration(gamma, env, v_diff,max_iter):
    n_states = env.env.nS
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    policy = np.zeros(n_states)
    V_prev = -1 * np.ones(n_states)
    k = 0
    while ((np.abs(V - V_prev)).max() > v_diff) & (k < max_iter):
        V_prev = V.copy()
        k += 1
        for s in range(n_states):
            act_val = np.zeros(n_actions)
            for a in range(n_actions):
                for T, s_, r, _ in env.P[s][a]:
                    act_val[a] += T * (r + gamma * V_prev[s_])
            V[s] = max(act_val)
            policy[s] = np.argmax(act_val)
    return policy, V, k


policy, V, k = value_iteration(gamma=0.9, env=env, v_diff=1e-10,max_iter=1000)
print(k)

#### Policy Iteration
def policy_eval(gamma, policy, env, V):
    n_states = env.observation_space.n
    policy_value = np.zeros(n_states)
    for s, a in enumerate(policy):
        for T, s_, r, _ in env.P[s][a]:
            policy_value[s] += T * (r + gamma * V[s_])
    return policy_value


def update_policy(gamma, V_prev, env, policy):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    for s in range(n_states):
        act_val = np.zeros(n_actions)
        for a in range(n_actions):
            for T, s_, r, _ in env.P[s][a]:
                act_val[a] += T * (r + gamma * V_prev[s_])
        policy[s] = np.argmax(act_val)
    return policy


def policy_iteration(gamma, env, max_iter, max_same_po):
    np.random.seed(42)
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    policy = np.random.randint(0, 4, n_states)
    policy_prev = np.zeros(n_states)
    i = 0
    for iter in range(max_iter):
        policy_prev = np.copy(policy)
        V = policy_eval(gamma, policy, env, V)
        policy = update_policy(gamma, V, env, policy)
        if np.all(np.equal(policy, policy_prev)):
            i += 1
        if i > max_same_po:
            break
    return policy, policy_prev, iter


policy_pi, policy_prev_pi, k_pi = policy_iteration(gamma=0.9, env=env, max_iter=1000, max_same_po=40)

env_large = env
#print(np.all(np.equal(policy_pi, policy.astype(int))))
###plot #1
k_vi_list = []
k_pi_list = []
time_vi_list = []
time_pi_list = []
convergence = []
max_same_n = [30, 50, 50,50,60, 60, 60]
i = 0
map_r = range(4, 32, 4)
for map_size in map_r:
    random_map = generate_random_map(size=map_size)
    env = gym.make("FrozenLake-v1", desc=random_map)
    ini_time = time.time()
    policy, V, k = value_iteration(gamma=0.9, env=env, v_diff=1e-10,max_iter = 10000)
    time_vi = time.time() - ini_time
    ini_time = time.time()
    policy_pi, policy_prev_pi, k_pi = policy_iteration(gamma=0.9, env=env, max_iter=1000,
                                                             max_same_po=max_same_n[i])
    time_pi = time.time() - ini_time
    convergence.append(np.all(np.equal(policy_pi, policy)))
    k_vi_list.append(k)
    k_pi_list.append(k_pi)
    time_vi_list.append(time_vi)
    time_pi_list.append(time_pi)
    i += 1

print(convergence)
state_r = [x**2 for x in map_r]
plt.plot(state_r, k_vi_list, label="Value Iteration")
plt.plot(state_r, k_pi_list, label="Policy Iteration")
plt.title("Iterations Needed for Convergence by # of states - gamma 0.9")
plt.xlabel('Number of Observation States')
plt.ylabel('Iteration Needed for Convergence')
plt.legend()

plt.plot(state_r, time_vi_list, label="Value Iteration")
plt.plot(state_r, time_pi_list, label="Policy Iteration")
plt.title("Run Time Spent for Convergence by # of states - gamma 0.9")
plt.xlabel('Number of Observation States')
plt.ylabel('Run Time Needed for Convergence')
plt.legend()
## plot #2 - varying grid size and gamma for the same policy

gamma_list=np.arange(0.6, 1.0, 0.1)
def policy_reward(algo, env,max_i_range, count, random,gamma_list=gamma_list):
    ave_rewards_full=[]
    for g in gamma_list:
        ave_rewards = []
        for max_i in max_i_range:
            if algo =='value':
                policy, _, _ = value_iteration(gamma=g, env=env, v_diff=1e-10,max_iter=max_i)
            elif algo == 'policy':
                policy_pi, _, _ = policy_iteration(gamma=g, env=env, max_iter=max_i,
                                                                   max_same_po=60)
                policy = policy_pi.copy()
            tot_reward = 0
            for n in range(count):
                done = False
                state = env.reset()
                while not done:
                    if random:
                        action = env.action_space.sample()
                    else:
                        action = policy[state]
                    state, reward, done, _ = env.step(action)
                    tot_reward += reward
            ave_rewards.append(tot_reward/count)
        ave_rewards_full.append(ave_rewards)
    return ave_rewards_full


#### run rewards results for the policy by different max number of iteration and gamma
env_large=env
map = generate_random_map(size=20, p=0.95)
env = gym.make("FrozenLake-v1", is_slippery=True,desc=map)
env = gym.make("FrozenLake-v1")
env=env_large
env.render()
env.reset()
ave_rewards_vi = policy_reward('value',env, range(1,160,10),200, False)

ave_rewards_pi =policy_reward('policy',env, range(1,160,10),200, False)

for i in range(0,len(ave_rewards_vi)):
    plt.plot(range(1,160,10),ave_rewards_vi[i],label="gamma = {}".format(round(gamma_list[i],2)))
plt.legend()
plt.title('Large Frozen Lake (400 states) Average Success Rate in 200 episodes by maximum training iteration and gamma - Value Iteration')
plt.xlabel('Maximum training iterations')
plt.ylabel('Average Success Rate')
for i in range(0,len(ave_rewards_pi)):
    plt.plot(range(1,160,10),ave_rewards_pi[i],label="gamma = {}".format(round(gamma_list[i],2)))
plt.legend()
plt.title('Large Frozen Lake (400 states) Average Success Rate in 200 episodes by maximum training iteration and gamma - Policy Iteration')
plt.xlabel('Maximum training iterations')
plt.ylabel('Average Success Rate')
##Q learner

# Training
def qlearner_train(env,train_episodes, step_max,epsilon, gamma, decay, alpha):
    env.reset()
    env._max_episode_steps=step_max
    Q_table = np.zeros((env.observation_space.n, env.action_space.n))
    times = []
    init_time = time.time()
    tot_step = []
    tot_rewards = []
    for episode in range(train_episodes):
        state = env.reset()
        done = False
        step=0
        rew = 0
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[state])
            state_new, reward, done, _ = env.step(action)
            Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (
                    reward + gamma * np.max(Q_table[state_new]))
            state = state_new
            rew += reward
            step+=1
        print('round {} - episode #{}'.format(train_episodes,episode))
        print('number of steps {}'.format(step))
        env.render()
        # epsilon = max(0.1, epsilon * decay)
        epsilon = max(0.0, epsilon - decay)
        tot_step.append(step)
        tot_rewards.append(rew)
    times.append((time.time() - init_time) * 1000)
    print(Q_table)
    rate =np.mean(tot_rewards)
    ave_step=np.mean(tot_step)
    print(rate)
    return Q_table,rate,times,ave_step

env.reset()
env.render()
episodes_count = 10000
max_step_range = range(1,10000,500)
qt_list_all=[]
reward_list_all=[]
time_list_all=[]
k_list_all=[]
eps_list=[0.7,0.95]
for epsilon in eps_list:
    env.reset()
    qt_list = []
    reward_rate_list = []
    time_list=[]
    k_list=[]
    decay = epsilon*0.0001
    for step_max in max_step_range:
        env.reset()
        Q_table,reward_rate, t,k= qlearner_train(env,train_episodes=episodes_count,step_max=step_max, gamma=0.9, epsilon=epsilon, decay=decay, alpha=0.9)
        reward_rate_list.append(reward_rate)
        time_list.append(t)
        qt_list.append(Q_table)
        k_list.append(k)
    qt_list_all.append(qt_list)
    time_list_all.append(time_list)
    reward_list_all.append(reward_rate_list)
    k_list_all.append(k_list)

for i in range(len(reward_list_all)):
    plt.plot(max_step_range,reward_list_all[i],label=('epsilon - {}'.format(eps_list[i])))
plt.legend()
plt.xlabel("Maximum step number in each episode")
plt.ylabel('Average Reward in {} Episodes'.format(episodes_count))
plt.title("Large Frozen Lake (400 States) QLearner - Success rate in 200 iterations by Maximum Step Number")

for i in range(len(k_list_all)):
    plt.plot(max_step_range,k_list_all[i],label=('epsilon - {}'.format(eps_list[i])))
plt.legend()
plt.xlabel("Maximum step number in each episode")
plt.ylabel('Average Number of Iterations in {} Episodes'.format(episodes_count))
plt.title("Large Frozen Lake (400 States) QLearner - Average Number of Steps Needed by Maximum Step Number")

for i in range(len(time_list_all)):
    plt.plot(max_step_range,time_list_all[i],label=('epsilon - {}'.format(eps_list[i])))
plt.legend()
plt.xlabel("Maximum step number in each episode")
plt.ylabel('Average Time Spent in {} Episodes'.format(episodes_count))
plt.title("Large Frozen Lake (400 States) QLearner - Total Time Spent by Maximum Step Number")

trained_qtable = qt_list_all[1][-2]
# Evaluation
nb_success = 0
rounds = 200
for _ in range(rounds):
    state = env.reset()
    env.render()
    done = False

    # Until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        action = np.argmax(trained_qtable[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = env.step(action)

        # Update our current state
        state = new_state

        # When we get a reward, it means we solved the game
        nb_success += reward
    env.render()

# Let's check our success rate!
print(f"Success rate = {nb_success / rounds * 100}%")

