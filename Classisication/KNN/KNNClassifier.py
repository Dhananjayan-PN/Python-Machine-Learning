import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('Week3/teleCust1000t.csv')
# print(df['custcat'].value_counts())
# df.hist(column='income', bins=50)
# plt.show()

X = df[[
    'region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ',
    'retire', 'gender', 'reside'
]].values
Y = df['custcat'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.1,
                                                    random_state=4)
# print('Train: ', X_train.shape, Y_train.shape)
# print('Test: ', X_test.shape, Y_test.shape)

# Startgin Classifier with k = 4
""" K = 4
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, Y_train)
predicted = knn.predict(X_test)
train_predicted = knn.predict(X_train)
 """

# Evaluating Accuracy
""" print('Train Accuracy:', metrics.accuracy_score(
    y_true=Y_train, y_pred=train_predicted))
print('Test Accuracy:', metrics.accuracy_score(
    y_true=Y_test, y_pred=predicted)) """

# Finding the best K
K_list = range(1, 601)
accuracies = []
for k in K_list:
    print(k, end=': ')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    predicted = knn.predict(X_test)
    train_predicted = knn.predict(X_train)
    accuracy = metrics.accuracy_score(y_true=Y_test, y_pred=predicted)
    print(accuracy)
    accuracies.append(accuracy)

print('Best K:', K_list[accuracies.index(max(accuracies))], 'with Accuracy:',
      max(accuracies))

# Plot the graph to visually find the value
plt.plot(K_list, accuracies, 'b-')
plt.show()
