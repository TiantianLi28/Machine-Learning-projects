#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    
    xTrain=np.array(xTrain)
    yTrain=np.array(yTrain)
    yTrain=np.ravel(yTrain)
    clf=GridSearchCV(KNeighborsClassifier(), {'n_neighbors':range(1,20,2)},cv=5,scoring='precision')
    clf.fit(xTrain,yTrain)
    print(clf.best_params_) 
    tree_param = {'max_depth': range(1,20,1),'min_samples_leaf': range(5,50),'criterion':['gini','entropy']}
    clf=GridSearchCV(DecisionTreeClassifier(),tree_param,cv=5,scoring='precision')
    clf.fit(xTrain,yTrain)
    print(clf.best_params_,clf.best_score_,clf.best_estimator_) 
    
    





if __name__ == "__main__":
    main()


