import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(0)
X, Y = make_blobs(n_samples=5000, centers=[
                  [4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# Visualize the data
# plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.show()

k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
k_means.fit(X)

# Get labels and centers of the clusters
labels = k_means.labels_
centers = k_means.cluster_centers_
print(labels, centers, sep='\n')

# Plot the clusters

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(len(centers)), colors):
    my_members = (labels == k)
    cluster_center = centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1],
            'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col,  markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()
