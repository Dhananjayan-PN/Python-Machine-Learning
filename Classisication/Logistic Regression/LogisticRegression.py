import pandas as pd
import numpy as np
import itertools
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

df = pd.read_csv("Week3/Logistic Regression/ChurnData.csv")
df["churn"] = df["churn"].astype("int")
# print(df.head())

X = df[[
    "tenure",
    "age",
    "address",
    "income",
    "ed",
    "employ",
    "equip",
    "callcard",
    "wireless",
    "longmon",
    "tollmon",
    "equipmon",
    "cardmon",
    "wiremon",
    "longten",
    "tollten",
    "cardten",
    "voice",
    "pager",
    "internet",
    "callwait",
    "confer",
    "ebill",
    "loglong",
    "logtoll",
    "lninc",
    "custcat",
]].values
X = preprocessing.StandardScaler().fit(X).transform(X)
Y = df["churn"].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4)

LogiRegr = LogisticRegression(C=0.01, solver="liblinear")
# C is the inverse of Regularization strength, a method used to avoid overfitting. Small positive float valus will result in well regularized models
# You could also use different solvers/optimizers like  ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

LogiRegr.fit(X_train, Y_train)
Y_predicted = LogiRegr.predict(X_test)

# ***Model Evaluation Methods***
# 1. Jaccard Index

jaccard_score = jaccard_similarity_score(Y_test, Y_predicted)
print("Jaccard Index: ", jaccard_score)


# 2. F-1 Score
F1 = f1_score(Y_test, Y_predicted)
print('F-1 Score: ', F1)


# 3. Log Loss
LogLoss = log_loss(Y_test, Y_predicted)
print('Log Loss: ', LogLoss)


# 4. Confusion Matrix
# Compute Confusion Matrix
cnf_matrix = confusion_matrix(Y_test, Y_predicted, labels=[1, 0])
print('Confusion Matrix:', cnf_matrix, sep='\n')

# Plot Confusion Matrix


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Showing Normalized confusion matrix")
    else:
        print("Showing Confusion matrix, without normalization")
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[
                      "churn=1", "churn=0"], normalize=False, title="Confusion matrix",)
plt.savefig('Week3/Logistic Regression/ConfusionMatrix.png')
plt.show()
