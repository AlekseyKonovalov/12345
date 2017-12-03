from sklearn.datasets import load_iris
from sklearn.tree import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X=data.data
y=data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

#print (X.shape)
#print (X_train.shape)
#print (X_test.shape)

#default tree
default_tree = DecisionTreeClassifier()
default_tree=default_tree.fit(X_train,y_train)

y_pred_train = default_tree.predict(X_train)
y_pred_test= default_tree.predict(X_test)

print("Accuracy of the training sample %s" %accuracy_score(y_train, y_pred_train))
print("Accuracy of the training sample %s" %accuracy_score(y_test, y_pred_test))
print('')

importances =default_tree.feature_importances_
print (importances)


#nodefault 1
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
classifier.fit(X_train,y_train)

y_pred_train = classifier.predict(X_train)
y_pred_test= classifier.predict(X_test)

print("Accuracy of the training sample %s" %accuracy_score(y_train, y_pred_train))
print("Accuracy of the training sample %s" %accuracy_score(y_test, y_pred_test))

importances =classifier.feature_importances_
print (importances)

#nodefault 2

for depth in range(1,10):
    tree_classifier = tree.DecisionTreeClassifier(
        max_depth=depth, random_state=0)

    classifier.fit(X_train,y_train)

    y_pred_train = classifier.predict(X_train)
    y_pred_test= classifier.predict(X_test)

    print("Accuracy of the training sample %s" %accuracy_score(y_train, y_pred_train))
    print("Accuracy of the training sample %s" %accuracy_score(y_test, y_pred_test))

    importances =classifier.feature_importances_
    print (importances)


