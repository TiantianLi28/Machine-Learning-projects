#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

def model_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTest = model.predict_proba(xTest)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    yHatTest = model.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return testAuc,testAcc

def model_train_test_auc(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    #yHatTest = model.predict_proba(xTest)
    yHatTest = model.predict(xTest)
    # calculate auc for test dataset
    print(len(yTest.columns),yHatTest[0])
    testAuc = roc_auc_score(yTest['label'],yHatTest)
    testAcc = 0#accuracy_score(yTest["label"], yHatTest)
    return testAuc,testAcc




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
    knnClass=KNeighborsClassifier(n_neighbors=15)
    dtClass=DecisionTreeClassifier(max_depth=3,min_samples_leaf=22)
    testAuc,testAcc=model_train_test(knnClass, xTrain, yTrain, xTest, yTest)
    testAucR,testAccR=model_train_test_auc(knnClass, xTrain, yTrain, xTest, yTest)
    DTtestAuc,DTtestAcc=model_train_test(dtClass, xTrain, yTrain, xTest, yTest)   
    
    x_train_copy=xTrain.copy()
    x_train_copy['y']=yTrain
    x_train_copy=shuffle(x_train_copy)
    x_train_test_shuffled=x_train_copy.iloc[:int(0.95*len(x_train_copy))]
    xTrain95=x_train_test_shuffled.iloc[:,0:(len(x_train_test_shuffled.columns)-1)]
    yTrain95=x_train_test_shuffled['y']
    print(len(xTrain95))
    testAuc95,testAcc95=model_train_test(knnClass, xTrain95, yTrain95, xTest, yTest)
    DTtestAuc95,DTtestAcc95=model_train_test(dtClass, xTrain95, yTrain95, xTest, yTest)     
    
    x_train_copy=xTrain.copy()
    x_train_copy['y']=yTrain
    x_train_copy=shuffle(x_train_copy)
    x_train_test_shuffled=x_train_copy.iloc[:int(0.9*len(x_train_copy))]
    xTrain90=x_train_test_shuffled.iloc[:,0:(len(x_train_test_shuffled.columns)-1)]
    yTrain90=x_train_test_shuffled['y']
    print(len(xTrain90))
    testAuc90,testAcc90=model_train_test(knnClass, xTrain90, yTrain90, xTest, yTest)
    DTtestAuc90,DTtestAcc90=model_train_test(dtClass, xTrain90, yTrain90, xTest, yTest)   
    
    x_train_copy=xTrain.copy()
    x_train_copy['y']=yTrain
    x_train_copy=shuffle(x_train_copy)
    x_train_test_shuffled=x_train_copy.iloc[:int(0.8*len(x_train_copy))]
    xTrain80=x_train_test_shuffled.iloc[:,0:(len(x_train_test_shuffled.columns)-1)]
    yTrain80=x_train_test_shuffled['y']
    print(len(xTrain80))
    testAuc80,testAcc80=model_train_test(knnClass, xTrain80, yTrain80, xTest, yTest)
    DTtestAuc80,DTtestAcc80=model_train_test(dtClass, xTrain80, yTrain80, xTest, yTest)     
    
    knnDF = pd.DataFrame([['100%', testAcc, testAuc],
                          ['100%', testAccR, testAucR],
                           ['95%', testAcc95, testAuc95],
                           ['90%', testAcc90, testAuc90],
                           ['80%', testAcc80, testAuc80]],
                           columns=['% training', 'test acc', 'test auc'])
    dtDF = pd.DataFrame([['100%', DTtestAcc, DTtestAuc],
                           ['95%', DTtestAcc95, DTtestAuc95],
                           ['90%', DTtestAcc90, DTtestAuc90],
                           ['80%', DTtestAcc80, DTtestAuc80],
                           ],
                           columns=['% training', 'test acc', 'test auc'])
    combined = pd.DataFrame(
                          [['100%', 'DT', DTtestAcc, DTtestAuc],
                           ['95%', 'DT', DTtestAcc95, DTtestAuc95],
                           ['90%', 'DT', DTtestAcc90, DTtestAuc90],
                           ['80%', 'DT', DTtestAcc80, DTtestAuc80],
                           ['100%', 'KNN',testAcc, testAuc],
                           ['95%', 'KNN',testAcc95, testAuc95],
                           ['90%', 'KNN',testAcc90, testAuc90],
                           ['80%', 'KNN',testAcc80, testAuc80]],
                           columns=['% training', 'classifier','test acc', 'test auc'])
    print(knnDF)
    print(dtDF)
    print(combined)













if __name__ == "__main__":
    main()
