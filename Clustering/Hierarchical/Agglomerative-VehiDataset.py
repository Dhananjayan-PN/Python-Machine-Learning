import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("Clustering\Hierarchical\cars_clus.csv")

# # Data cleansing
df[[
    'sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas',
    'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg', 'lnsales'
]] = df[[
    'sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas',
    'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg', 'lnsales'
]].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
featureset = df[[
    'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt',
    'fuel_cap', 'mpg'
]]
x = featureset.values  #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)

Agglo = AgglomerativeClustering(n_clusters=6, linkage="complete")
Agglo.fit(feature_mtx)
# print(Agglo.labels_)

df["Cluster"] = Agglo.labels_
# print(df.head)

n_clusters = max(Agglo.labels_) + 1
colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))
agg_cars = df.groupby(['Cluster', 'type'])['horsepow', 'engine_s', 'mpg',
                                           'price'].mean()

plt.figure(figsize=(16, 10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label, ), ]
    for i in subset.index:
        plt.text(
            subset.loc[i][0] + 5, subset.loc[i][2], 'type=' + str(int(i)) +
            ', price=' + str(int(subset.loc[i][3])) + 'k')
    plt.scatter(subset.horsepow,
                subset.mpg,
                s=subset.price * 20,
                c=color,
                label='cluster' + str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()
