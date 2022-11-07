import time
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_samples, silhouette_score,calinski_harabasz_score,davies_bouldin_score,homogeneity_score,\
    completeness_score,v_measure_score,mean_squared_error,plot_confusion_matrix,f1_score
from sklearn.preprocessing import (StandardScaler, LabelEncoder)
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
from sklearn.decomposition import PCA,FastICA,TruncatedSVD
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from scipy.stats import  kurtosis
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
rn = 2
####Neural Network
df1= pd.read_csv("./data/voice.csv")

df1.label = [1 if each == "male" else 0 for each in df1.label]
##heatmap and correlation
plt.figure(figsize=(18, 14))
pd.DataFrame(abs(df1.corr()['label'].drop('label') * 100).sort_values(ascending=False)).plot.bar(figsize=(15, 12))

df1['label'].value_counts()

##build modeling dataset
x = df1.drop(['label'], axis=1).values
y = df1['label'].values
##split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

##handle outlier (scale data)
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

## Euclidean distance is the default distance in sklearn
kmeans_kwargs = {
    'init': 'k-means++',
    'n_init': 10,
    'max_iter': 300,
    # 'random_state':42,
    #   'algorithm':'lloyd',
}
n_range = range(2, 15)

def generate_kmeans_curve(n_range, kmeans_kwargs,x, y):
    sse_all = []
    sc_all = []
    vm_all = []
    chs_all = []
    dbs_all = []
    for seed in [42, 1105]:
        sse = []
        sc = []
        vm = []
        chs = []
        dbs = []
        for k in n_range:
            clusterer = KMeans(n_clusters=k, random_state=seed, **kmeans_kwargs)
            clusterer.fit(x)
            sse.append(clusterer.inertia_)
            labels = clusterer.fit_predict(x)
            sc.append(silhouette_score(x, labels, metric='euclidean', sample_size=None, random_state=seed))
            chs.append(calinski_harabasz_score(x, clusterer.labels_))
            dbs.append(davies_bouldin_score(x, clusterer.labels_))
            vm.append(v_measure_score(y, clusterer.labels_))
        sse_all.append(sse)
        sc_all.append(sc)
        vm_all.append(vm)
        chs_all.append(chs)
        dbs_all.append(dbs)
        kl = KneeLocator(
            n_range, sse_all[0], curve='convex', direction='decreasing'
        )
        print(kl.elbow)
    return {
        'sse_all': sse_all,
        'sc_all': sc_all,
        'chs_all':chs_all,
        'dbs_all':dbs_all,
        'vm_all':vm_all
    }
kmeans_res = generate_kmeans_curve(n_range, kmeans_kwargs,x_train,y_train )
# plot
def plot_kmeans_gmm_n_clusters(data_name,red_name,kmeans_res,gmm_res):
    fig,axes = plt.subplots(nrows = 1,ncols=5)
    axe = axes.ravel()
    metrics = ['sse_all','sc_all','chs_all','dbs_all','vm_all']
    metrics_name = ['Sum of Squared Error','Silhouette Score','Calinski Harabasz Score','Davies Bouldin Score','V-Measure']

    for i in range(0,len(metrics)):
        ax = axe[i]
        ax.plot(n_range,kmeans_res[metrics[i]][0],label = 'seed1')
        ax.plot(n_range,kmeans_res[metrics[i]][1], label='seed2',)
        ax.set_title(metrics_name[i])
        ax.xaxis.set_ticks(range(2,15,2))
        ax.legend()
    fig.suptitle("{} Cluster Evaluation Metrics - {} - Kmeans".format(data_name, red_name))

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=5)
    axe = axes.ravel()
    metrics = ['bic_all', 'sc_all', 'chs_all', 'dbs_all', 'vm_all']
    metrics_name = ['BIC', 'Silhouette Score', 'Calinski Harabasz Score', 'Davies Bouldin Score',
                    'V-Measure']
    for i in range(0, len(metrics)):
        ax = axe[i]
        ax.plot(n_range, gmm_res[metrics[i]][0], label=covariance_type[0])
        ax.plot(n_range, gmm_res[metrics[i]][1], label=covariance_type[1])
        ax.plot(n_range, gmm_res[metrics[i]][2], label=covariance_type[2])
        ax.plot(n_range, gmm_res[metrics[i]][3], label=covariance_type[3])
        ax.set_title(metrics_name[i])
        ax.xaxis.set_ticks(range(2, 15, 2))
        ax.legend()
    fig.suptitle("{} Cluster Evaluation Metrics - {} - GMM".format(data_name,red_name))

n_components = range(2, 15)
covariance_type = ['spherical', 'tied', 'diag', 'full']
def generate_gmm_curves(n_components,covariance_type,x,y):
    bic_all = []
    sc_all = []
    vm_all = []
    chs_all = []
    dbs_all = []
    for cov in covariance_type:
        bic = []
        sc = []
        vm = []
        chs = []
        dbs = []
        for n_comp in n_components:
            clusterer = GaussianMixture(n_components=n_comp, covariance_type=cov)
            clusterer.fit(x)
            labels = clusterer.fit_predict(x)
            bic.append(clusterer.bic(x))
            sc.append(silhouette_score(x, labels, metric='euclidean', sample_size=None, random_state=rn))
            chs.append(calinski_harabasz_score(x, labels))
            dbs.append(davies_bouldin_score(x, labels))
            vm.append(v_measure_score(y, labels))
        bic_all.append(bic)
        sc_all.append(sc)
        vm_all.append(vm)
        chs_all.append(chs)
        dbs_all.append(dbs)
    return {
        'bic_all': bic_all,
        'sc_all': sc_all,
        'chs_all':chs_all,
        'dbs_all':dbs_all,
        'vm_all':vm_all
    }
gmm_res=generate_gmm_curves(n_components,covariance_type,x_train,y_train)

plot_kmeans_gmm_n_clusters("Bank Attrition","Original",kmeans_res,gmm_res)

#####validation of cluster

kmeans_test = generate_kmeans_curve([3], kmeans_kwargs, x_test,y_test)

gmm_test =generate_gmm_curves([2],covariance_type,x_test,y_test)

def generate_silhouette_scores(range_n_clusters, cluster_name, x, plot_attr_1, plot_attr_2):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if cluster_name == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=rn)
        if cluster_name == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=rn,covariance_type='diag')


        cluster_labels = clusterer.fit_predict(x)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(x, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(x, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            x[:, plot_attr_1], x[:, plot_attr_2], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )
        if cluster_name == 'kmeans':
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, plot_attr_1],
                centers[:, plot_attr_2],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[plot_attr_1], c[plot_attr_2], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the {} feature".format(plot_attr_1+1))
        ax2.set_ylabel("Feature space for the {} feature".format(plot_attr_2+1))

        plt.suptitle(
            "Silhouette analysis for {} clustering on Bank Attrition data with n_clusters = %d".format(cluster_name)
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

generate_silhouette_scores([3], 'kmeans', x_train, 1, 6)
generate_silhouette_scores([2], 'gmm', x_train, 1, 6)
#validate results
clusterer = KMeans(n_clusters=3, random_state=rn).fit(x_test)
y_test_labels = 1 - y_test
sum(y_test_labels == clusterer.labels_)/len(y_test_labels)

######dimension reduction####
###PCA
losses=[]
pca_n = x_train.shape[1]
pca_20 = PCA(pca_n).fit(x_train)
for i in range(1,pca_n+1):
    pca = PCA(n_components=i)
    pca.fit(x_train)
    X_train_pca = pca.transform(x_train)
    X_projected = pca.inverse_transform(X_train_pca)
    loss = np.sum((x_train - X_projected) ** 2, axis=1).mean()
    losses.append(loss)
fig, axs = plt.subplots(1, 2)
axs[0].plot(range(1, pca_n+1), np.cumsum(pca_20.explained_variance_ratio_*100), marker='o')
axs[0].xaxis.set_ticks(range(1, pca_n+1, 2))
axs[0].set(xlabel='# of Component', ylabel='Cumulative Explained Variance Ratio')
axs[1].plot(range(1, pca_n+1), losses, marker='o')
axs[1].xaxis.set_ticks(range(1, pca_n+1, 2))
axs[1].set(xlabel='# of Components', ylabel='Re-construction error')

fig.suptitle('Principle Component Analysis - Bank Attrition')
fig = plt.gcf()
fig.set_size_inches(16, 7)

# PCA transform
pca = PCA(6).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
x_test_pca = pca.transform(x_test)
# plot transformed data
plt.figure()
plt.scatter(x=x_train_pca[:,0], y=x_train_pca[:,1],c=y_train)

####fit NN
from sklearn.neural_network import MLPClassifier

nn_pca= MLPClassifier(
          solver='lbfgs',
          activation='tanh',
          hidden_layer_sizes=(1,))
nn_pca.fit(x_train_pca,y_train)
plot_confusion_matrix(nn_pca,
                      x_test_pca,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])
print(f1_score(y_test,nn_pca.predict(x_test_pca)))
# kmeans and gmm clustering after dimension reduction
pca_kmeans_res = generate_kmeans_curve(n_range, kmeans_kwargs,x_train_pca,y_train )
gmm_pca_res=generate_gmm_curves(n_components,covariance_type,x_train_pca,y_train)

plot_kmeans_gmm_n_clusters("Bank Attrition","PCA",pca_kmeans_res,gmm_pca_res)

# test clustering
generate_silhouette_scores([3], 'kmeans', x_test_pca, 0, 1)

####fit NN after dimension reduction
def add_cluster_label_nn(dim_red_type,n_cluster,n_component,cov_type,train_data_reduced,test_data_reduced,y_train,y_test):
    kmeans = KMeans(n_clusters=n_cluster, random_state=rn, **kmeans_kwargs)
    kmeans_train =kmeans.fit(train_data_reduced)
    x_train_km = np.hstack((train_data_reduced,np.atleast_2d(kmeans_train.labels_).T))
    kmeans_test = kmeans.fit(test_data_reduced)
    x_test_km = np.hstack((test_data_reduced,np.atleast_2d(kmeans_test.labels_).T))

    nn_km= MLPClassifier(
              solver='lbfgs',
              activation='tanh',
              hidden_layer_sizes=(1,))
    nn_km.fit(x_train_km,y_train)
    km_score = f1_score(y_test,nn_km.predict(x_test_km))
    print("{} + Kmeans cluster info + NN testing F1 Score - {}".format(dim_red_type,km_score))

    #gmm
    gmm = GaussianMixture(n_components=n_component,random_state=rn,covariance_type=cov_type)
    gmm_train_label = gmm.fit_predict(train_data_reduced)
    x_train_gmm = np.hstack((train_data_reduced,np.atleast_2d(gmm_train_label).T))
    gmm_test_label = gmm.fit_predict(test_data_reduced)
    x_test_gmm = np.hstack((test_data_reduced,np.atleast_2d(gmm_test_label).T))

    st=time.time()
    nn_gmm= MLPClassifier(
              solver='lbfgs',
              activation='tanh',
              hidden_layer_sizes=(1,))
    nn_gmm.fit(x_train_gmm,y_train)
    print(time.time() - st)
    gmm_score = f1_score(y_test, nn_km.predict(x_test_gmm))
    print("{} + GMM cluster info + NN testing F1 Score - {}".format(dim_red_type,gmm_score))
    return [km_score,
            gmm_score]

add_cluster_label_nn("PCA",3,2,'diag',x_train_pca,x_test_pca,y_train,y_test)
# ICA
losses = []
kur = []

for i in range(1, 25):
    transformer = FastICA(n_components=i, random_state=rn)
    X_transformed = transformer.fit_transform(x_train)
    kur.append(np.mean(np.abs(kurtosis(X_transformed))))
    X_projected = transformer.inverse_transform(X_transformed)
    loss = np.sum((x_train - X_projected) ** 2, axis=1).mean()
    losses.append(loss)

fig, axs = plt.subplots(1, 2)

axs[0].plot(range(1, 25), losses, marker='o')
axs[0].set(xlabel='# of Components', ylabel='Re-construction error')
axs[0].xaxis.set_ticks(range(1, 25, 2))
axs[1].plot(range(1, 25), kur, marker='o')
axs[1].set(xlabel='# of Components', ylabel='Average Kurtosis')
axs[1].xaxis.set_ticks(range(1, 25, 2))
fig = plt.gcf()
fig.suptitle('Independent Components Analysis - Bank Attrition')
fig.set_size_inches(16, 7)

transformer = FastICA(n_components=6, random_state=rn)
X_transformed = transformer.fit_transform(x_train)
X_transformed_test = transformer.fit_transform(x_test)
# plot transformed data
plt.figure()
plt.scatter(x=X_transformed[:,0], y=X_transformed[:,1],c=y_train)

####fit NN
nn_ica= MLPClassifier(
          solver='lbfgs',
          activation='tanh',
          hidden_layer_sizes=(1,))
nn_ica.fit(X_transformed,y_train)
plot_confusion_matrix(nn_ica,
                      X_transformed_test,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])
print(f1_score(y_test,nn_ica.predict(X_transformed_test)))
# kmeans clustering after dimension reduction
ica_kmeans_res = generate_kmeans_curve(n_range, kmeans_kwargs,X_transformed,y_train )
# EM clustering after ICA
gmm_ica_res=generate_gmm_curves(n_components,covariance_type,X_transformed,y_train)
#plot
plot_kmeans_gmm_n_clusters("Bank Attrition","ICA",ica_kmeans_res,gmm_ica_res)

# cluster info + NN
add_cluster_label_nn("ICA",3,2,'diag',X_transformed,X_transformed_test,y_train,y_test)
##########################
#randomized projection
losses=[]

for i in range(1,25):
    random_projection = SparseRandomProjection(n_components=i)
    random_projection.fit(x_train)
    components =  random_projection.components_.toarray() # shape=(5, 11)
    p_inverse = np.linalg.pinv(components.T) # shape=(5, 11)
    #now get the transformed data using the projection components
    reduced_data = random_projection.transform(x_train) #shape=(4898, 5)
    reconstructed= reduced_data.dot(p_inverse)  #shape=(4898, 11)
    error = mean_squared_error(x_train, reconstructed)
    losses.append(error)

plt.plot(range(1,25),losses,marker='o')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("# Components", fontsize=18)
plt.ylabel("Re-construction error", fontsize=18)

###
rp_seeds_res = []
iter_n = range(1,105,5)
for seed in iter_n:
    random_projection = SparseRandomProjection(n_components=6,random_state=seed)
    random_projection.fit(x_train)
    x_transformed = random_projection.transform(x_train)
    x_test_transformed=random_projection.transform(x_test)
    # NN
    nn_rp= MLPClassifier(
              solver='lbfgs',
              activation='tanh',
              hidden_layer_sizes=(1,))
    nn_rp.fit(x_transformed,y_train)
    test_score = f1_score(y_test,nn_rp.predict(x_test_transformed))
    print(test_score)
    rp_seeds_res.append(test_score)

plt.plot(iter_n,rp_seeds_res)
plt.xticks(iter_n)
plt.axhline(y=np.mean(rp_seeds_res), color='r', linestyle='-')
plt.xlabel('Random Seed')
plt.ylabel('Neural Network Testing F1 Score')
plt.title("Randomized Projection Testing Performance By Random States - Voice")
# kmeans clustering after RP
rp_kmeans_res = generate_kmeans_curve(n_range, kmeans_kwargs,x_transformed,y_train )
# EM clustering after RP
rp_gmm_res=generate_gmm_curves(n_range,covariance_type,x_transformed,y_train)
# plot
plot_kmeans_gmm_n_clusters("Bank Attrition","RP",rp_kmeans_res,rp_gmm_res)
##cluster + dim red + NN
rp_cluster_seeds_res = []
for seed in iter_n:
    random_projection = SparseRandomProjection(n_components=6,random_state=seed)
    random_projection.fit(x_train)
    x_transformed = random_projection.transform(x_train)
    x_test_transformed=random_projection.transform(x_test)
    rp_cluster_seeds_res.append(add_cluster_label_nn("RP",2,2,'diag',x_transformed,x_test_transformed,y_train,y_test))

km_test = [x[0] for x in rp_cluster_seeds_res]
gmm_test = [x[1] for x in rp_cluster_seeds_res]

plt.plot(iter_n,km_test,label='Kmeans')
plt.plot(iter_n,gmm_test,label='GMM')
plt.xlabel('random seed')
plt.ylabel('Testing F1 Score')
plt.legend()
plt.title("Randomized Projection Performance with Cluster Label By Random States - Voice")
#####################################
## Truncated SVD
tsvd_full = TruncatedSVD(20).fit(x_train)

plt.figure()
plt.plot(range(1, 21), np.cumsum(tsvd_full.explained_variance_ratio_*100), marker='o')
plt.xticks(range(1, 21, 2))


tsvd = TruncatedSVD(6).fit(x_train)
x_transformed = tsvd.transform(x_train)
x_test_transformed=tsvd.transform(x_test)

# NN
st = time.time()
nn_tsvd= MLPClassifier(
          solver='lbfgs',
          activation='tanh',
          hidden_layer_sizes=(1,))
nn_tsvd.fit(x_transformed,y_train)
print(time.time() - st)
plot_confusion_matrix(nn_tsvd,
                      x_test_transformed,
                      y_test,
                      values_format='d',
                      display_labels=['No Attri', 'Attr'])
print(f1_score(y_test,nn_tsvd.predict(x_test_transformed)))

# kmeans clustering after TSVD
tsvd_kmeans_res = generate_kmeans_curve(n_range, kmeans_kwargs,x_transformed,y_train )
gmm_tsvd_res=generate_gmm_curves(n_range,covariance_type,x_transformed,y_train)
plot_kmeans_gmm_n_clusters("Bank Attrition","TSVD",tsvd_kmeans_res,gmm_tsvd_res)

# cluster + dim red + NN
add_cluster_label_nn("TSVD",3,2,'diag',x_transformed,x_test_transformed,y_train,y_test)