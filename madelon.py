import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

def calculateAccuracyScore(algorithm, X_train, y_train, X_test, y_test):
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)
    return accuracy_score(y_test, y_pred) 

def selectFeatures(algorithm, X_train, y_train, numberOfFeatures, isForward):
    selector = sfs(algorithm, k_features=(1,numberOfFeatures), forward=isForward, floating=False, verbose=0, scoring='accuracy', cv=None, n_jobs=-1)
    selector.fit(X_train.values, y_train) 
    return list(selector.k_feature_idx_)

def predictWithSelection(algorithm, X_train, y_train, X_test, y_test, numberOfFeatures=5, isForward=True):
    selectionType = "Forward Selection" if isForward else "Backward Selection"
    print("%s-> Selecting features with %s..." % (selectionType, algorithm.__class__.__name__))
    
    feat_cols = selectFeatures(algorithm, X_train, y_train, numberOfFeatures, isForward)

    print("%s completed" % selectionType)
    print("Selected features (indexes):", feat_cols)
    
    X_train = X_train[X_train.columns[feat_cols]]
    X_test = X_test[X_test.columns[feat_cols]]

    print("New Feature Count: ", len(X_train.columns))
    print('Training accuracy on selected features: %.3f' % calculateAccuracyScore(LinearDiscriminantAnalysis(), X_train, y_train, X_train, y_train))
    print('Testing accuracy on selected features: %.3f' % calculateAccuracyScore(LinearDiscriminantAnalysis(), X_train, y_train, X_test, y_test))
    print("********************")

madelonDir = './madelon/MADELON/'

df = pd.read_csv(madelonDir + 'madelon_train.data', sep=' ', header=None).drop(500, axis=1)
df_labels = pd.read_csv(madelonDir + 'madelon_train.labels', sep=' ', header=None)

df_test = pd.read_csv(madelonDir + 'madelon_valid.data', sep=' ', header=None).drop(500, axis=1)
df_test_labels = pd.read_csv(madelonDir + 'madelon_valid.labels', sep=' ', header=None)

y_train = df_labels[0]
X_train = df

y_test = df_test_labels[0]
X_test = df_test

models = [
    KNeighborsClassifier(n_neighbors=1),
    KNeighborsClassifier(n_neighbors=2),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=4),
    KNeighborsClassifier(n_neighbors=5),
    LogisticRegression(solver='liblinear'),
    LinearDiscriminantAnalysis(),
    tree.DecisionTreeClassifier(),
]

print("Madelon Dataset")
print("----------------")

print("\n*Without feature selection*\n")
print("Feature count: ", len(X_train.columns))
print("Accuracy scores")
print("----------------")
for model in models:
    print("%s: %.3f" % (model.__class__.__name__, calculateAccuracyScore(model, X_train, y_train, X_test, y_test)))
    print("----------------")

print("\n*With Feature Selection*\n")
for model in models:
    predictWithSelection(model, X_train, y_train, X_test, y_test, numberOfFeatures=10, isForward=True)

for model in models:
    predictWithSelection(model, X_train, y_train, X_test, y_test, numberOfFeatures=5, isForward=False)