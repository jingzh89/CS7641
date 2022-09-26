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
import timeit
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

df1= pd.read_csv("./data/voice.csv")

df1.label = [1 if each == "male" else 0 for each in df1.label]
##heatmap and correlation
plt.figure(figsize=(18, 14))
sns.heatmap(df1.corr(), annot=True, cmap="YlGnBu")
pd.DataFrame(abs(df1.corr()['label'].drop('label') * 100).sort_values(ascending=False)).plot.bar(figsize=(15, 12))

df1['label'].value_counts()

##build modeling dataset
x = df1.drop(['label'], axis=1).values
y = df1['label'].values

##split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

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
    cross_scores = cross_val_score(clf_df, x_train, y_train, scoring ='accuracy', cv=5)
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
dtree_clf = DecisionTreeClassifier(random_state=1105,ccp_alpha=0.000816)
start = timeit.default_timer()
dtree_clf.fit(x_train,
            y_train)
stop = timeit.default_timer()
print('Training Time: ', stop - start)
start = timeit.default_timer()
dtree_clf.predict(x_train)
stop = timeit.default_timer()
print('Testing Time: ', stop - start)
#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=dtree_clf,
    title='Accuracy for Decision Tree - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0.93, 1.001),
    cv=5,
    n_jobs=10,
    scoring="accuracy",
)

#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(dtree_clf, x_train, y_train, scoring='accuracy',cv=5,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for Decision Tree Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('Accuracy Score')

dtree_clf.score(x_test,y_test)

# plot tree
plt.figure(figsize=(15, 7.5))
plot_tree(dtree_clf,
          filled=True,
          rounded=True,
          class_names=['No Attri', 'Yes Attri'],
          feature_names=mdf.drop(['Attrition'], axis=1).columns)
# plot confusion matrix
plot_confusion_matrix(dtree_clf, x_test, y_test)
y_train_pred = dtree_clf.predict(x_train)
y_pred = dtree_clf.predict(x_test)

print('training accuracy {}'.format(dtree_clf.score(x_train,y_train)))
print('testing accuracy {}'.format(dtree_clf.score(x_test,y_test)))

##XGboost
## parameter tuning
param_grid = {
    'max_depth': [6,7,8,9],
    'learning_rate': [0.09,0.1,1.25,1.5],
    'gamma': [0.15,0.175,0.2,0.25],
    'reg_lambda': [0.5,1,3]
}

optimal_params = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',
                                                          seed=1105),
                              param_grid=param_grid,
                              scoring="accuracy",
                              verbose=2,
                              n_jobs=10,
                              cv=5)
optimal_params.fit(x_train,
                   y_train,
                   #early_stopping_rounds=10,
                   #eval_set=[(x_test,y_test)],
                   verbose=False)

print(optimal_params.best_params_)

clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            max_depth=8,
                            gamma=0.2,
                            learning_rate=0.1,
                            reg_lambda=1,
                            missing=0, seed=1105)
start = timeit.default_timer()
clf_xgb.fit(x_train,
            y_train)
stop = timeit.default_timer()
print('Training Time: ', stop - start)
start = timeit.default_timer()
clf_xgb.predict(x_train)
stop = timeit.default_timer()
print('Training Time: ', stop - start)

clf_xgb.score(x_test,y_test)
plot_confusion_matrix(clf_xgb,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])

#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=clf_xgb,
    title='Accuracy for XGBoost - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0.95, 1.001),
    cv=5,
    n_jobs=10,
    scoring="accuracy",
)


y_train_pred = clf_xgb.predict(x_train)
y_pred = clf_xgb.predict(x_test)
print('training accuracy {}'.format(clf_xgb.score(x_train,y_train)))
print('testing accuracy {}'.format(clf_xgb.score(x_test,y_test)))

#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf_xgb, x_train, y_train, scoring='accuracy',cv=5,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for XGBoost Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('Accuracy Score')
#plt.close()

# KNN
from sklearn.neighbors import KNeighborsClassifier
param_grid = {
    'n_neighbors': [1,2,3,4,5],
}

optimal_params = GridSearchCV(estimator=KNeighborsClassifier(),
                              param_grid=param_grid,
                              scoring="accuracy",
                              n_jobs=10,
                              cv=5)
optimal_params.fit(x_train,
                   y_train)

print(optimal_params.best_params_)

knn = KNeighborsClassifier(n_neighbors=1)
start = timeit.default_timer()
knn.fit(x_train,y_train)
stop = timeit.default_timer()
print('Training Time: ', stop - start)
start = timeit.default_timer()
knn.predict(x_train)
stop = timeit.default_timer()
print('Testing Time: ', stop - start)

plot_confusion_matrix(knn,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])

#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=knn,
    title='Accuracy for KNN - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0.8, 1.05),
    cv=5,
    n_jobs=10,
    scoring="accuracy",
)


y_train_pred = knn.predict(x_train)
y_pred = knn.predict(x_test)

#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(knn, x_train, y_train, scoring='accuracy',cv=5,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for KNN Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('Accuracy Score')

print('training accuracy {}'.format(knn.score(x_train,y_train)))
print('testing accuracy {}'.format(knn.score(x_test,y_test)))

#SVC

from sklearn.svm import SVC
param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
optimal_params = GridSearchCV(estimator=SVC(),
                              param_grid = param_grid,
                              scoring="accuracy",
                              n_jobs=10,
                              cv=5
                              )
optimal_params.fit(x_train,y_train)

print(optimal_params.best_params_)

svm = SVC(C=0.1,
          degree=1,
          gamma=1,
          kernel='poly')
start = timeit.default_timer()
svm.fit(x_train,y_train)
stop = timeit.default_timer()
print('Training Time: ', stop - start)
start = timeit.default_timer()
svm.predict(x_train)
stop = timeit.default_timer()
print('Testing Time: ', stop - start)

plot_confusion_matrix(svm,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])
#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=svm,
    title='Accuracy for SVM - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0.95, 0.98),
    cv=5,
    n_jobs=10,
    scoring="accuracy",
)
y_train_pred = svm.predict(x_train)
y_pred = svm.predict(x_test)

#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(svm, x_train, y_train, scoring='accuracy',cv=5,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for SVM Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('Accuracy Score')

print('training accuracy {}'.format(svm.score(x_train,y_train)))
print('testing accuracy {}'.format(svm.score(x_test,y_test)))

#nueral network
from sklearn.neural_network import MLPClassifier
param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
             ]
        }
       ]

optimal_params = GridSearchCV(estimator=MLPClassifier(),
                              param_grid = param_grid,
                              scoring="accuracy",
                              n_jobs=10,
                              cv=5
                              )
optimal_params.fit(x_train,y_train)

print(optimal_params.best_params_)

nn = MLPClassifier(
          solver='lbfgs',
          activation='relu',
          hidden_layer_sizes=(10,))

start = timeit.default_timer()
nn.fit(x_train,y_train)
stop = timeit.default_timer()
print('Training Time: ', stop - start)
start = timeit.default_timer()
nn.predict(x_train)
stop = timeit.default_timer()
print('Testing Time: ', stop - start)


plot_confusion_matrix(nn,
                      x_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])
#plot training curve
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(
    estimator=nn,
    title='Accuracy for Neural Network - Training/Cross Validation',
    X=x_train,
    y=y_train,
    axes=axes[:, 0],
    ylim=(0.93, 1.001),
    cv=5,
    n_jobs=10,
    scoring="accuracy",
)

#train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(nn, x_train, y_train, scoring='accuracy',cv=5,return_times=True)
#plt.plot(train_sizes,np.mean(train_scores,axis=1),label='Training')
#plt.plot(train_sizes,np.mean(test_scores,axis=1),label = 'Validation')
#plt.title('Learning Curve for Neural Network Classifier - Training and Cross Validation')
#plt.legend()
#plt.xlabel('Sample Size')
#plt.ylabel('Accuracy Score')
y_train_pred = nn.predict(x_train)
y_pred = nn.predict(x_test)
print('training accuracy {}'.format(nn.score(x_train,y_train)))
print('testing accuracy {}'.format(nn.score(x_test,y_test)))

