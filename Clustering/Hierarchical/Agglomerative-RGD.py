import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

X, Y = make_blobs(n_samples=50, centers=[
    [4, 4], [-2, -1], [1, 1], [10, 4]], cluster_std=0.9)
# plt.scatter(X[:, 0], X[:, 1], marker='o')
# plt.show()

Agglo = AgglomerativeClustering(n_clusters=4, linkage='average')
Agglo.fit(X, Y)

# Plot the clusters
plt.figure(figsize=(6, 4))
x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
X = (X - x_min) / (x_max - x_min)
for i in range(X.shape[0]):
    plt.text(X[i, 0], X[i, 1], str(Y[i]),
             color=plt.cm.nipy_spectral(Agglo.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()


dist_matrix = distance_matrix(X, X)
# print(dist_matrix)

Z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)
