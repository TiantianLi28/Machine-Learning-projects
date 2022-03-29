'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import time

 
def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    timeStart=time.time()
    testProp=int((1-testSize)*len(xFeat))
    xFeatTrain=xFeat.iloc[:testProp]
    xFeatTest=xFeat.iloc[testProp:]
    yTrueTrain=y.iloc[:testProp]
    yTrueTest=y.iloc[testProp:]
    model.fit(xFeatTrain,yTrueTrain)
    yHatTrain=model.predict_proba(xFeatTrain)
    yHatTest=model.predict_proba(xFeatTest)
    trainAuc = roc_auc_score(yTrueTrain,yHatTrain[:, 1])
    testAuc = roc_auc_score(yTrueTest,yHatTest[:, 1])
    timeEnd=time.time()
    timeElapsed = timeEnd-timeStart
    # TODO fill int
    return trainAuc, testAuc, timeElapsed


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set. 


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    timeStart=time.time()
    xFeatLength=len(xFeat)
    prop=float(1/k)
    start=0
    trainAuc=0.0
    testAuc=0.0
    while start+prop<=1:
        xFeatTrainLeft=xFeat.iloc[0:int(start*xFeatLength)]
        xFeatTest=xFeat.iloc[int(start*xFeatLength):int((start+prop)*xFeatLength)]
        xFeatTrainRight=xFeat.iloc[int((start+prop)*xFeatLength):]
        xFeatTrain1=[xFeatTrainLeft,xFeatTrainRight]
        xFeatTrain=pd.concat(xFeatTrain1)
        
        yTrueTrainLeft=y.iloc[0:int(start*xFeatLength)]
        yTrueTest=y.iloc[int(start*xFeatLength):int((start+prop)*xFeatLength)]
        yTrueTrainRight=y.iloc[int((start+prop)*xFeatLength):]
        yTrueTrain1=[yTrueTrainLeft,yTrueTrainRight]    
        yTrueTrain=pd.concat(yTrueTrain1)
        
        model.fit(xFeatTrain,yTrueTrain)
        yHatTrain=model.predict_proba(xFeatTrain)
        yHatTest=model.predict_proba(xFeatTest)
        
        trainAuc += roc_auc_score(yTrueTrain,yHatTrain[:, 1])
        testAuc += roc_auc_score(yTrueTest,yHatTest[:, 1])
        
        start+=prop

    trainAuc = float(trainAuc/k)
    testAuc = float(testAuc/k)
    timeEnd=time.time()
    timeElapsed = timeEnd-timeStart
    # TODO FILL IN
    return trainAuc, testAuc, timeElapsed


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    
    trainAuc = 0.0
    testAuc = 0.0
    timeElapsed = 0.0
    startTime=time.time()
    # TODO FILL IN
    for i in range(s):
        xFeat_copy=xFeat.copy()
        xFeat_copy['label']=y
        #shuffle the dataframe and pick the first k samples, which reaches the same effect as randomly choosing k smaples
        xFeat_copy=shuffle(xFeat_copy)
        trainA,testA,time1=holdout(model, xFeat_copy.iloc[:,0:(len(xFeat_copy.columns)-2)], xFeat_copy['label'], testSize)
        trainAuc+=trainA
        testAuc+=testA
    endTime=time.time()
    timeElapsed=endTime-startTime
    return trainAuc/s, testAuc/s, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
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
    print(xTrain.columns)
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


def main():
    # set up the program to take in arguments from the command line
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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.70)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.70, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.70, 10)
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)

    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
