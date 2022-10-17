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

####Neural Network
da = pd.read_csv("./data/Employee_Attrition.csv")

# encode categorical variables
df = da.copy()
Attri_le = LabelEncoder()
BusTravel_le = LabelEncoder()
Department_le = LabelEncoder()
EduField_le = LabelEncoder()
Gender_le = LabelEncoder()
JobRole_le = LabelEncoder()
MaritalStatus_le = LabelEncoder()
Over18_le = LabelEncoder()
OverTime_le = LabelEncoder()

df['Attrition'] = Attri_le.fit_transform(df['Attrition'])
df['BusinessTravel'] = BusTravel_le.fit_transform(df['BusinessTravel'])
df['Department'] = Department_le.fit_transform(df['Department'])
df['EducationField'] = Department_le.fit_transform(df['EducationField'])
df['Gender'] = Gender_le.fit_transform(df['Gender'])
df['JobRole'] = JobRole_le.fit_transform(df['JobRole'])
df['MaritalStatus'] = MaritalStatus_le.fit_transform(df['MaritalStatus'])
df['Over18'] = Over18_le.fit_transform(df['Over18'])
df['OverTime'] = OverTime_le.fit_transform(df['OverTime'])

##target variable
mdf = df[['Attrition', 'OverTime', 'TotalWorkingYears', 'Age', 'JobLevel', 'YearsInCurrentRole', 'MaritalStatus',
          'MonthlyIncome', 'YearsWithCurrManager', 'YearsAtCompany', 'StockOptionLevel', 'JobInvolvement',
          'JobSatisfaction', 'EnvironmentSatisfaction', 'DistanceFromHome', 'Department', 'WorkLifeBalance', 'JobRole',
          'DailyRate', 'RelationshipSatisfaction', 'TrainingTimesLastYear', ]]

##build modeling dataset
x = mdf.drop(['Attrition'], axis=1).values
y = mdf['Attrition'].values

##split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1105, stratify=y)

##handle outlier (scale data)
x_train = RobustScaler().fit_transform(x_train)
x_test = RobustScaler().fit_transform(x_test)

rhc_nn = []
ga_nn = []
sa_nn = []

for length in range(10, 2000, 200):
    time_start = time.time()
    nn_sa = mlrose.NeuralNetwork(hidden_nodes=(1,), activation='tanh',
                                 algorithm='simulated_annealing',
                                 max_iters=length, random_state=1,early_stopping=True,learning_rate=0.1,
                                  clip_max=5, max_attempts=100, bias=True, is_classifier=True,
                                 curve=True)

    nn_sa.fit(x_train, y_train)
    fit_time = time.time()
    print(f'fit_time = {fit_time - time_start}')
    y_train_pred = nn_sa.predict(x_train)
    y_test_pred = nn_sa.predict(x_test)
    y_train_f1 = f1_score(y_train, y_train_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    sa_nn.append([fit_time - time_start, y_train_f1, y_test_f1])

    time_start = time.time()
    nn_rhc = mlrose.NeuralNetwork(hidden_nodes=(1,), activation='tanh',
                                  algorithm='random_hill_climb',
                                  max_iters=length, bias=True, random_state=1,early_stopping=True,learning_rate=0.1,
                                  clip_max=5, max_attempts=100, is_classifier=True, curve=True)

    nn_rhc.fit(x_train, y_train)
    fit_time = time.time()
    print(f'fit_time = {fit_time - time_start}')
    y_train_pred = nn_rhc.predict(x_train)
    y_test_pred = nn_rhc.predict(x_test)
    y_train_f1 = f1_score(y_train, y_train_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    rhc_nn.append([fit_time - time_start, y_train_f1, y_test_f1])

    time_start = time.time()
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=(1,), activation='tanh',
                                 algorithm='genetic_alg',random_state=1,early_stopping=True,learning_rate=0.1,
                                  clip_max=5, max_attempts=100,
                                 max_iters=length, bias=True, is_classifier=True, curve=True)

    nn_ga.fit(x_train, y_train)
    fit_time = time.time()
    print(f'fit_time = {fit_time - time_start}')
    y_train_pred = nn_ga.predict(x_train)
    y_test_pred = nn_ga.predict(x_test)
    y_train_f1 = f1_score(y_train, y_train_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    ga_nn.append([fit_time - time_start, y_train_f1, y_test_f1])

problem_name = 'Neural Network'
metrics_labels = ['Time', 'Train F1', 'Test F1']
metrics_nn = []
for i in range(0, len(metrics_labels)):
    df = pd.concat([pd.DataFrame(rhc_nn)[i], pd.DataFrame(ga_nn)[i], pd.DataFrame(sa_nn)[i]], ignore_index=True, axis=1)
    df.index = range(10, 1000, 100)
    df.columns = ['RHC', 'GA', 'SA']
    metrics_nn.append(df)

fig, axes = plt.subplots(nrows=1, ncols=3)
axe = axes.ravel()
for j in range(0, len(metrics_labels), 1):
    metrics_nn[j].plot(marker='.', xlabel="Problem Size", ylabel=metrics_labels[j],
                       title="{}".format(metrics_labels[j]), ax=axe[j])
# fig.subplots_adjust()
fig.suptitle('Performance Metrics for {}'.format(problem_name))

## learning rate - 0.01

rhc_nn = []
ga_nn = []
sa_nn = []

for length in range(10, 2000, 200):
    time_start = time.time()
    nn_sa = mlrose.NeuralNetwork(hidden_nodes=(1,), activation='tanh',
                                 algorithm='simulated_annealing',
                                 max_iters=length, random_state=1,early_stopping=True,
                                  clip_max=5, max_attempts=100, bias=True, is_classifier=True,learning_rate=0.01,
                                 curve=True)

    nn_sa.fit(x_train, y_train)
    fit_time = time.time()
    print(f'fit_time = {fit_time - time_start}')
    y_train_pred = nn_sa.predict(x_train)
    y_test_pred = nn_sa.predict(x_test)
    y_train_f1 = f1_score(y_train, y_train_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    sa_nn.append([fit_time - time_start, y_train_f1, y_test_f1])

    time_start = time.time()
    nn_rhc = mlrose.NeuralNetwork(hidden_nodes=(1,), activation='tanh',
                                  algorithm='random_hill_climb',
                                  max_iters=length, bias=True, random_state=1,early_stopping=True,learning_rate=0.01,
                                  clip_max=5, max_attempts=100, is_classifier=True, curve=True)

    nn_rhc.fit(x_train, y_train)
    fit_time = time.time()
    print(f'fit_time = {fit_time - time_start}')
    y_train_pred = nn_rhc.predict(x_train)
    y_test_pred = nn_rhc.predict(x_test)
    y_train_f1 = f1_score(y_train, y_train_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    rhc_nn.append([fit_time - time_start, y_train_f1, y_test_f1])

    time_start = time.time()
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=(1,), activation='tanh',
                                 algorithm='genetic_alg',random_state=1,early_stopping=True,learning_rate=0.01,
                                  clip_max=5, max_attempts=100,
                                 max_iters=length, bias=True, is_classifier=True, curve=True)

    nn_ga.fit(x_train, y_train)
    fit_time = time.time()
    print(f'fit_time = {fit_time - time_start}')
    y_train_pred = nn_ga.predict(x_train)
    y_test_pred = nn_ga.predict(x_test)
    y_train_f1 = f1_score(y_train, y_train_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)
    ga_nn.append([fit_time - time_start, y_train_f1, y_test_f1])

problem_name = 'Neural Network'
metrics_labels = ['Time', 'Train F1', 'Test F1']
metrics_nn = []
for i in range(0, len(metrics_labels)):
    df = pd.concat([pd.DataFrame(rhc_nn)[i], pd.DataFrame(ga_nn)[i], pd.DataFrame(sa_nn)[i]], ignore_index=True, axis=1)
    df.index = range(10, 1000, 100)
    df.columns = ['RHC', 'GA', 'SA']
    metrics_nn.append(df)

fig, axes = plt.subplots(nrows=1, ncols=3)
axe = axes.ravel()
for j in range(0, len(metrics_labels), 1):
    metrics_nn[j].plot(marker='.', xlabel="Problem Size", ylabel=metrics_labels[j],
                       title="{}".format(metrics_labels[j]), ax=axe[j])
# fig.subplots_adjust()
fig.suptitle('Performance Metrics for {}'.format(problem_name))

