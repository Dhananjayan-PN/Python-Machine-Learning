import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('Week4\Partitioned-based\Cust_Segmentation.csv')
df = df.drop('Address', axis=1)
# print(df.head())

X = np.nan_to_num(df.values[:, 1:])
X = StandardScaler().fit_transform(X)

# Modeling
k = 3
k_means = KMeans(init="k-means++", n_clusters=k, n_init=12)
k_means.fit(X)
labels = k_means.labels_
centers = k_means.cluster_centers_
# print(labels, centers, sep='\n')

# Assign a new of labels
df["Cluster"] = labels
df.groupby("Cluster").mean()
# print(df.head())

# Visualize the clusters
# 2D Plot
# area = np.pi * (X[:, 1])**2
# plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
# plt.xlabel('Age', fontsize=18)
# plt.ylabel('Income', fontsize=16)
# plt.show()

# 3D PLot
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))
plt.show()
