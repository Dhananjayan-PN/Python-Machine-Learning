import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pydotplus
import graphviz
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn import tree

df = pd.read_csv('Classification/Decision Trees/drug200.csv')
# print(df.head(20))
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
Y = df['Drug'].values

label_sex = preprocessing.LabelEncoder()
label_sex.fit(['F', 'M'])
X[:, 1] = label_sex.transform(X[:, 1])

label_bp = preprocessing.LabelEncoder()
label_bp.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = label_bp.transform(X[:, 2])

label_chol = preprocessing.LabelEncoder()
label_chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = label_chol.transform(X[:, 3])

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=3)

# print('Train:', X_train.shape, Y_train.shape)
# print('Test:', X_test.shape, Y_test.shape)

Tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
Tree.fit(X_train, Y_train)

Y_predicted = Tree.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_true=Y_test, y_pred=Y_predicted))

# Visualize the tree
dot_data = StringIO()
filename = "Classification/Decision Trees/DecisionTree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out = tree.export_graphviz(Tree,
                           feature_names=featureNames,
                           out_file=dot_data,
                           class_names=np.unique(Y_train),
                           filled=True,
                           special_characters=True,
                           rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
plt.show()
