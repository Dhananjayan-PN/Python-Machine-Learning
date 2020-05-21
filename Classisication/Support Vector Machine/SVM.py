import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

df = pd.read_csv('Classification\Support Vector Machine\cell_samples.csv')
# print(df.head())

# Visualize the data
""" ax = df[df['Class'] == 4][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
df[df['Class'] == 2][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
plt.show() """

# print(df.dtypes)

df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')

X = df[[
    'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc',
    'BlandChrom', 'NormNucl', 'Mit'
]].values
Y = df['Class'].values

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=4)

# Check dimensions of datasets
# print("Train: ", X_train.shape, Y_train.shape)
# print("Test: ", X_test.shape, Y_test.shape)

SVMclf = svm.SVC(kernel='poly')
# kernel can also be 'linear', 'rbf', 'poly', 'sigmoid'. Default = 'rbf'
SVMclf.fit(X_train, Y_train)

Y_predicted = SVMclf.predict(X_test)

# Model Evaluation

# Jaccard Index/Score
jaccard_index = jaccard_similarity_score(Y_test, Y_predicted)
print('Jaccard Index:', jaccard_index)

# F-1 Score
f1 = f1_score(Y_test, Y_predicted, pos_label=2)
print('F-1 Score:', f1)

# Confusion Matrix

# Compute
ConfMatrix = confusion_matrix(Y_test, Y_predicted, labels=[2, 4])
print('Confusion Matrix:', ConfMatrix, sep='\n')
np.set_printoptions(precision=2)

# Plot


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Showing Normalized confusion matrix")
    else:
        print("Showing Confusion matrix, without normalization")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.figure()
plot_confusion_matrix(
    ConfMatrix,
    classes=["Benign(2)", "Malignant(4)"],
    normalize=False,
    title="Confusion matrix",
)
plt.savefig('Classification/Support Vector Machine/ConfusionMatrix.png')
plt.show()
