import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df = pd.read_csv('Final Project -  The Best Classifier\loan_train.csv')
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['dayofweek'] = df['effective_date'].dt.dayofweek
df['loan_status'].value_counts()
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)
df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis=1, inplace=True)

X = Feature
y = df['loan_status'].values

X = preprocessing.StandardScaler().fit(X).transform(X)

# KNN Classification

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=4)

K_list = range(1, 100)
accuracies = []
for k in K_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)
    train_predicted = knn.predict(X_train)
    accuracy = accuracy_score(y_true=y_test, y_pred=predicted)
    accuracies.append(accuracy)

plt.plot(K_list, accuracies, 'b-')
plt.show()

print('\nBest K for KNN:', K_list[accuracies.index(max(accuracies))],
      'with Accuracy:', max(accuracies))
k = K_list[accuracies.index(max(accuracies))]
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Decision Tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=4)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dt.fit(X_train, y_train)
predicted = dt.predict(X_test)
print('Decision Tree Accuracy: ', accuracy_score(y_test, predicted))

# Support Vector Machine
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df['loan_status'].replace(to_replace=['COLLECTION', 'PAIDOFF'],
                          value=[0, 1],
                          inplace=True)
df['loan_status'] = df['loan_status'].astype('int')
y = df['loan_status'].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=4)
SVM = svm.SVC(kernel='poly', gamma='auto')
SVM.fit(X_train, y_train)
predicted = SVM.predict(X_test)
print('SVM Accuracy: ', accuracy_score(y_test, predicted))

# Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
y = df['loan_status'].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=4)
lr = LogisticRegression(C=0.01, solver="lbfgs")
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print('Logistic regression Accuracy: ', accuracy_score(y_test, predicted))

# Evaluation
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

test_df = pd.read_csv('Final Project -  The Best Classifier\loan_test.csv')
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)
test_df['Gender'].replace(to_replace=['male', 'female'],
                          value=[0, 1],
                          inplace=True)
Features = test_df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
Features = pd.concat([Features, pd.get_dummies(test_df['education'])], axis=1)
Features.drop(['Master or Above'], axis=1, inplace=True)
Xtest = Features
Xtest = preprocessing.StandardScaler().fit(Xtest).transform(Xtest)
ytest = test_df['loan_status'].values

predicted = knn.predict(Xtest)
print('\nKNN:')
print('Jaccard Index:', jaccard_similarity_score(ytest, predicted))
print('F1-score:', f1_score(ytest, predicted, pos_label='PAIDOFF'))

predicted = dt.predict(Xtest)
print('\nDecision Tree:')
print('Jaccard Index:', jaccard_similarity_score(ytest, predicted))
print('F1-score:', f1_score(ytest, predicted, pos_label='PAIDOFF'))

predicted = SVM.predict(Xtest)
test_df['loan_status'].replace(to_replace=['COLLECTION', 'PAIDOFF'],
                               value=[0, 1],
                               inplace=True)
test_df['loan_status'] = test_df['loan_status'].astype('int')
ytest = test_df['loan_status'].values
print('\nSVM:')
print('Jaccard Index:', jaccard_similarity_score(ytest, predicted))
print('F1-score:', f1_score(ytest, predicted))

predicted = lr.predict(Xtest)
ytest = test_df['loan_status'].values
print('\nLogistic Regression:')
print('Jaccard Index:', jaccard_similarity_score(ytest, predicted))
print('F1-score:', f1_score(ytest, predicted))
print('Log Loss:', log_loss(ytest, predicted))
