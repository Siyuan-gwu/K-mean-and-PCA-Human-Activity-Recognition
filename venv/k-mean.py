import random
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

data = pd.read_csv('train.csv')

print (data.sample(5))

print (str(data.shape))

labels = data['activity']
data = data.drop(['rn', 'activity'], axis = 1)
labels_keys = labels.unique().tolist()
labels = np.array(labels)
print('Activity labels: ' + str(labels_keys))

# min-max normalization
def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
data = MaxMinNormalization(data)
print (data.sample(5))

# check the optimal k value
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(data)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()


def k_means(n_clust, data_frame, true_labels):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels
    and clustering performance parameters.

    Input:
    n_clust - number of clusters (k value)
    data_frame - dataset we want to cluster
    true_labels - original labels

    Output:
    1 - crosstab of cluster and actual labels
    2 - performance table
    """
    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (k_means.inertia_,
             homogeneity_score(true_labels, y_clust),
             completeness_score(true_labels, y_clust),
             v_measure_score(true_labels, y_clust),
             adjusted_rand_score(true_labels, y_clust),
             adjusted_mutual_info_score(true_labels, y_clust),
             silhouette_score(data_frame, y_clust, metric='euclidean')))

k_means(2, data, labels)

#change labels to binary
#0 - not moving, 1 - moving
labels_binary = labels.copy()
for i in range(len(labels_binary)):
    if (labels_binary[i] == 'STANDING' or labels_binary[i] == 'SITTING' or labels_binary[i] == 'LAYING'):
        labels_binary[i] = 0
    else:
        labels_binary[i] = 1
labels_binary = np.array(labels_binary.astype(int))

k_means(2, data, labels_binary)

#PCA
pca = PCA(random_state=123)
pca.fit(data)
features = range(pca.n_components_)

plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='blue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()

def pca_transform(n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global data_reduced
    data_reduced = pca.fit_transform(data)
    print('Shape of the new Data df: ' + str(data_reduced.shape))

pca_transform(1)
k_means(2, data_reduced, labels_binary)

pca_transform(2)
k_means(2, data_reduced, labels_binary)

