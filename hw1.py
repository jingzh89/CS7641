import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (train_test_split, GridSearchCV, learning_curve, ShuffleSplit)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (roc_curve, auc, roc_auc_score,f1_score,
                             confusion_matrix, plot_confusion_matrix,classification_report)

import xgboost as xgb

import matplotlib.pyplot as plt

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

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

####EDA
df.Attrition.value_counts()
sns.countplot(x= df['Attrition'])


##heatmap and correlation
plt.figure(figsize=(18, 14))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
pd.DataFrame(abs(df.corr()['Attrition'].drop('Attrition') * 100).sort_values(ascending=False)).plot.bar(figsize=(15, 12))

##target variable
mdf = df[['Attrition', 'OverTime', 'TotalWorkingYears', 'Age', 'JobLevel', 'YearsInCurrentRole', 'MaritalStatus',
          'MonthlyIncome', 'YearsWithCurrManager', 'YearsAtCompany', 'StockOptionLevel', 'JobInvolvement',
          'JobSatisfaction', 'EnvironmentSatisfaction', 'DistanceFromHome', 'Department', 'WorkLifeBalance', 'JobRole',
          'DailyRate', 'RelationshipSatisfaction', 'TrainingTimesLastYear', ]]
mdf['Attrition'].value_counts()
sns.countplot(data=mdf, x='Attrition')
plt.legend(labels=['Yes', 'No'])

numCols = mdf.select_dtypes([np.number]).columns
numCols
for col in numCols:
    plt.figure(figsize=(18,6))
    sns.displot(x=col,data=mdf, palette="mako", hue='Attrition')
    plt.show()

numCols = mdf.select_dtypes([np.number]).columns
numCols
#for col in numCols:
#    plt.figure(figsize=(18,6))
#    sns.displot(x=col,data=mdf, palette="mako", hue='Attrition')
#    plt.show()

##build modeling dataset
x = mdf.drop(['Attrition'], axis=1).values
y = mdf['Attrition'].values

##split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1105, stratify=y)

##handle outlier (scale data)
x_train = RobustScaler().fit_transform(x_train)
x_test = RobustScaler().fit_transform(x_test)

# pruning
dtree_clf = DecisionTreeClassifier(random_state=1105)
path = dtree_clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
ccp_alphas = ccp_alphas[:-1]
alpha_loop_values = []
for ccp_alpha in ccp_alphas:
    clf_df = DecisionTreeClassifier(random_state=1105, ccp_alpha=ccp_alpha)
    cross_scores = cross_val_score(clf_df, x_train, y_train, scoring ='f1', cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(cross_scores), np.std(cross_scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')
print(alpha_results.sort_values('mean_accuracy',ascending=False))
## decision tree classifier
dtree_clf = DecisionTreeClassifier(random_state=1105,ccp_alpha=0.004564)
#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=dtree_clf,
    title='F1 Score for Decision Tree - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0.0, 1.25),
    cv=3,
    n_jobs=4,
    scoring="f1",
)
#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(dtree_clf, x_train, y_train, scoring='f1',cv=3,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for Decision Tree Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('F1 Score')
#plt.close()

dtree_clf.fit(x_train, y_train)
y_train_pred = dtree_clf.predict(x_train)
y_pred = dtree_clf.predict(x_test)

# report F1 score
classification_report(y_train,y_train_pred)
f1_score(y_test,y_pred)
classification_report(y_test,y_pred)
# plot tree
plt.figure(figsize=(15, 7.5))
plot_tree(dtree_clf,
          filled=True,
          rounded=True,
          class_names=['No Attri', 'Yes Attri'],
          feature_names=mdf.drop(['Attrition'], axis=1).columns)
# plot confusion matrix
plot_confusion_matrix(dtree_clf, x_test, y_test)
print('training f1 {}'.format(f1_score(y_train,y_train_pred)))
print('testing f1 {}'.format(f1_score(y_test,y_pred)))
##XGboost
## parameter tuning
param_grid = {
    'max_depth': [4, 5,6,7],
    'learning_rate': [0.025,0.05,0.1],
    'gamma': [0.25, 1.0,1.25],
    'reg_lambda': [5,10.0, 20, 100],
    'scale_pos_weight': [5]
}

optimal_params = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',
                                                          seed=1105),
                              param_grid=param_grid,
                              scoring="f1",
                              verbose=2,
                              n_jobs=10,
                              cv=3)
optimal_params.fit(x_train,
                   y_train,
                   #early_stopping_rounds=10,
                   #eval_set=[(x_test,y_test)],
                   verbose=False)

print(optimal_params.best_params_)

#f1

clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            max_depth=6,
                            gamma=1,
                            learning_rate=0.05,
                            reg_lambda=20,
                            scale_pos_weight=5.2,
                            missing=0, seed=1105)
clf_xgb.fit(x_train,
            y_train)

plot_confusion_matrix(clf_xgb,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])

#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=clf_xgb,
    title='F1 Score for XGBoost Tree - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0.0, 1),
    cv=3,
    n_jobs=10,
    scoring="f1",
)
#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf_xgb, x_train, y_train, scoring='f1',cv=3,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for XGBoost Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('F1 Score')
#plt.close()
y_train_pred = clf_xgb.predict(x_train)
y_pred = clf_xgb.predict(x_test)
print('training f1 {}'.format(f1_score(y_train,y_train_pred)))
print('testing f1 {}'.format(f1_score(y_test,y_pred)))
#plot grid search iteration
max_depth_range = [1,2,3,4,5,6,7]
lambda_range = [1,5,10,15,20,25,30]
grid_search_plot = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',
                                                          seed=1105),
                                param_grid={
                                    'max_depth': max_depth_range,
                                    'learning_rate': [0.05],
                                    'gamma': [1.0],
                                    'reg_lambda': lambda_range,
                                    'scale_pos_weight': [5.2]
                                },
                              scoring="f1",
                              verbose=2,
                              n_jobs=10,
                              cv=3)
grid_search_plot.fit(x_train,y_train)

scores = grid_search_plot.cv_results_['mean_test_score']
scores = np.array(scores).reshape(len(max_depth_range), len(lambda_range))

plt.figure(figsize=(12,8))
for ind, i in enumerate(max_depth_range):
    plt.plot(lambda_range, scores[ind], label='Depth: ' + str(i))
plt.legend()
plt.xlabel('reg_lambda')
plt.ylabel('Mean F1 score')
plt.title('XGBoost Grid Search Scores in term of Max Depth and Lambda', fontsize=16)
plt.show()
# KNN
from sklearn.neighbors import KNeighborsClassifier
param_grid = {
    'n_neighbors': [1,2,3,4,5],
}

optimal_params = GridSearchCV(estimator=KNeighborsClassifier(),
                              param_grid=param_grid,
                              scoring="f1",
                              n_jobs=10,
                              cv=3)
optimal_params.fit(x_train,
                   y_train)

print(optimal_params.best_params_)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
plot_confusion_matrix(knn,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])

#plot training curve

y_train_pred = knn.predict(x_train)
y_pred = knn.predict(x_test)
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=knn,
    title='F1 Score for KNN - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0, 0.8),
    cv=3,
    n_jobs=10,
    scoring="f1",
)
#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(knn, x_train, y_train, scoring='f1',cv=3,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for KNN Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('F1 Score')
print('training f1 {}'.format(f1_score(y_train,y_train_pred)))
print('testing f1 {}'.format(f1_score(y_test,y_pred)))



#SVC

from sklearn.svm import SVC
param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
optimal_params = GridSearchCV(estimator=SVC(),
                              param_grid = param_grid,
                              scoring="f1",
                              n_jobs=10,
                              cv=3
                              )
optimal_params.fit(x_train,y_train)

print(optimal_params.best_params_)
svm = SVC(C=100,
          degree=1,
          gamma=0.01,
          kernel='rbf')
svm.fit(x_train,y_train)
plot_confusion_matrix(svm,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])
#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=svm,
    title='F1 Score for SVM - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0, 1),
    cv=3,
    n_jobs=10,
    scoring="f1",
)
#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(svm, x_train, y_train, scoring='f1',cv=3,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for SVM Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('F1 Score')
y_train_pred = svm.predict(x_train)
y_pred = svm.predict(x_test)
print('testing f1 {}'.format(f1_score(y_test,y_pred)))
print('training f1 {}'.format(f1_score(y_train,y_train_pred)))

#nueral network
from sklearn.neural_network import MLPClassifier
param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(1,1),(1,2),(1,1,1),
             ]
        }
       ]

optimal_params = GridSearchCV(estimator=MLPClassifier(),
                              param_grid = param_grid,
                              scoring="f1",
                              n_jobs=10,
                              cv=3
                              )
optimal_params.fit(x_train,y_train)

print(optimal_params.best_params_)

nn = MLPClassifier(
          solver='lbfgs',
          activation='tanh',
          hidden_layer_sizes=(1,))
nn.fit(x_train,y_train)
plot_confusion_matrix(nn,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])
#plot training curve

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=nn,
    title='F1 Score for Neural Network - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0, 1),
    cv=3,
    n_jobs=10,
    scoring="f1",
)
#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(nn, x_train, y_train, scoring='f1',cv=3,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for Neural Network Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('F1 Score')
y_train_pred = nn.predict(x_train)
y_pred = nn.predict(x_test)
print('testing f1 {}'.format(f1_score(y_test,y_pred)))
print('training f1 {}'.format(f1_score(y_train,y_train_pred)))