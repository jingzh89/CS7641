import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (train_test_split, GridSearchCV, learning_curve, ShuffleSplit)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (roc_curve, auc, roc_auc_score, f1_score,
                             confusion_matrix, plot_confusion_matrix, classification_report)

np.random.seed(2)


def random_optimization(problem_name, length_lower, length_upper, length_interval, max_attempt):
    rhc_res = []
    ga_res = []
    sa_res = []
    mimic_res = []
    for length in range(length_lower, length_upper, length_interval):
        print(length)
        if problem_name == 'knap sack':
            weights = np.random.uniform(low=0.1, high=1, size=(length,))
            # weights = [10, 5, 2, 8, 15]
            values = np.random.uniform(low=1, high=length, size=(length,))
            # values = [1, 2, 3, 4, 5]
            fitness = mlrose.Knapsack(weights, values)
        if problem_name == 'queens':
            fitness = mlrose.Queens()
        if problem_name == 'four peaks':
            fitness = mlrose.FourPeaks(t_pct=.5)
        if problem_name == 'flip flop':
            fitness = mlrose.FlipFlop()
        if problem_name == 'max k color':
            fitness = mlrose.MaxKColor([(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)])
        if problem_name == 'continuous peak':
            fitness = mlrose.ContinuousPeaks(t_pct=0.15)
        if problem_name == 'one max':
            fitness = mlrose.FlipFlop()

        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)

        start_time = time.time()
        _, _, curve = mlrose.random_hill_climb(problem, restarts=10 * length, max_attempts=max_attempt,
                                               max_iters=length * 10,
                                               init_state=None, curve=True, random_state=1)
        end_time = time.time()
        rhc_res.append([end_time - start_time, curve[-1, 0], curve[-1, 1], curve[-1, 1] / (end_time - start_time)])

        start_time = time.time()
        _, _, curve = mlrose.genetic_alg(problem, pop_size=10 * length, mutation_prob=0.4, max_attempts=max_attempt,
                                         max_iters=length * 10, curve=True, random_state=1)
        end_time = time.time()
        ga_res.append(
            [end_time - start_time, curve[-1, 0], curve[-1, 1], curve[-1, 1] / (end_time - start_time)])

        start_time = time.time()
        _, _, curve = mlrose.simulated_annealing(problem, schedule=mlrose.GeomDecay(0.95), max_attempts=max_attempt,
                                                 init_state=None,
                                                 max_iters=length * 10, curve=True)
        end_time = time.time()
        sa_res.append(
            [end_time - start_time, curve[-1, 0], curve[-1, 1], curve[-1, 1] / (end_time - start_time)])

        start_time = time.time()
        _, _, curve = mlrose.mimic(problem, pop_size=10 * length, keep_pct=0.2, max_attempts=max_attempt,
                                   max_iters=length * 10,
                                   curve=True, random_state=1)
        end_time = time.time()
        mimic_res.append([end_time - start_time, curve[-1, 0], curve[-1, 1],
                          curve[-1, 1] / (end_time - start_time)])
    return {
        'rhc': rhc_res,
        'ga': ga_res,
        'sa': sa_res,
        'mimic': mimic_res
    }


def plot_metric_size(problem_name, res, length_lower, length_upper, length_interval):
    metrics = ['Time', 'Best Fitness', 'Function Eval', 'Eval per sec']
    metrics_res = []
    for i in range(0, len(metrics), 1):
        df = pd.concat([pd.DataFrame(res['ga'])[i], pd.DataFrame(res['sa'])[i],
                        pd.DataFrame(res['mimic'])[i], pd.DataFrame(res['rhc'])[i]], ignore_index=True, axis=1)
        df.index = range(length_lower, length_upper, length_interval)
        df.columns = ["GA", "SA", "MIMIC", "RHC"]
        metrics_res.append(df)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axe = axes.ravel()
    for j in range(0, len(metrics), 1):
        metrics_res[j].plot(marker='.', xlabel="Problem Size", ylabel=metrics[j],
                            title="{}".format(metrics[j]), ax=axe[j])
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Performance Metrics for {}'.format(problem_name))


length_lower = 10
length_upper = 110
length_interval = 10

# four peaks (GA)
problem_name = 'four peaks'
max_attempt = 10
fourpeaks = random_optimization(problem_name, length_lower, length_upper, length_interval, max_attempt)
plot_metric_size(problem_name, fourpeaks, length_lower, length_upper, length_interval)

# continuous peak
problem_name = 'continuous peak'
max_attempt = 100
length_upper = 100
continuouspeak = random_optimization(problem_name, length_lower, length_upper, length_interval, max_attempt)
plot_metric_size(problem_name, continuouspeak, length_lower, length_upper, length_interval)

# kcolor
kcolor = random_optimization('max k color', length_lower, length_upper, length_interval, max_attempt)
plot_metric_size('max k color', kcolor, length_lower, length_upper, length_interval)

# flipflop (MIMIC)
problem_name = 'flip flop'
max_attempt = 10
flipflop = random_optimization(problem_name, length_lower, length_upper, length_interval, max_attempt)
plot_metric_size(problem_name, flipflop, length_lower, length_upper, length_interval)

# queens (MIMIC)
problem_name = 'queens'
max_attempt = 10
flipflop = random_optimization(problem_name, length_lower, length_upper, length_interval, max_attempt)
plot_metric_size(problem_name, flipflop, length_lower, length_upper, length_interval)

# knapsack (MIMIC)
problem_name = 'knap sack'
max_attempt = 6
knapsack = random_optimization(problem_name, length_lower, length_upper, length_interval, max_attempt)
plot_metric_size(problem_name, knapsack, length_lower, length_upper, length_interval)

# Count One
problem_name = 'one max'
max_attempt = 100
length_upper = 100
onemax = random_optimization(problem_name, length_lower, length_upper, length_interval, max_attempt)
plot_metric_size(problem_name, onemax, length_lower, length_upper, length_interval)
