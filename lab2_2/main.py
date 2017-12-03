import numpy as np
from sklearn.tree import tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import graphviz
import pydotplus
import collections

dataset = np.loadtxt("data_banknote_authentication.txt", delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:3]
y = dataset[:,4]

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

X_train= dataset[0:1029,0:3]
X_test=dataset[1029:1372,0:3]

y_train=dataset[0:1029,4]
y_test=dataset[1029:1372,4]

#print (X.shape)
print (X_train.shape)
print (X_test.shape)

#default tree
default_tree = DecisionTreeClassifier()
default_tree=default_tree.fit(X_train,y_train)

y_pred_train = default_tree.predict(X_train)
y_pred_test= default_tree.predict(X_test)

print("Accuracy of the training sample %s" %accuracy_score(y_train, y_pred_train))
print("Accuracy of the testing sample %s" %accuracy_score(y_test, y_pred_test))

scores = cross_val_score(estimator=default_tree, X=X, y=y, scoring='accuracy', cv=5)
print (scores)
print (scores.mean())

#dot_data=export_graphviz(default_tree, out_file='1.txt', filled=True, rounded=True, special_characters=True)
#graph=graphviz.Source(dot_data)
#graph

importances =default_tree.feature_importances_
print (importances)
print('')


#nodefault tree 1
nodefault_tree1 = DecisionTreeClassifier(criterion='entropy')
nodefault_tree1=nodefault_tree1.fit(X_train,y_train)

y_pred_train = nodefault_tree1.predict(X_train)
y_pred_test= nodefault_tree1.predict(X_test)

print("Accuracy of the training sample %s" %accuracy_score(y_train, y_pred_train))
print("Accuracy of the testing sample %s" %accuracy_score(y_test, y_pred_test))

scores = cross_val_score(estimator=nodefault_tree1, X=X, y=y, scoring='accuracy', cv=5)
print (scores)
print (scores.mean())

importances =nodefault_tree1.feature_importances_
print (importances)
print('')

#nodefault tree 2
nodefault_tree2 = DecisionTreeClassifier(max_depth=15, splitter='random')
nodefault_tree2=nodefault_tree2.fit(X_train,y_train)

y_pred_train = nodefault_tree2.predict(X_train)
y_pred_test= nodefault_tree2.predict(X_test)

print("Accuracy of the training sample %s" %accuracy_score(y_train, y_pred_train))
print("Accuracy of the testing sample %s" %accuracy_score(y_test, y_pred_test))
scores = cross_val_score(estimator=nodefault_tree2, X=X, y=y, scoring='accuracy', cv=5)
print (scores)
print (scores.mean())

importances =nodefault_tree2.feature_importances_
print (importances)

print('')


