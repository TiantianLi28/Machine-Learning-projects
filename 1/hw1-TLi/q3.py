# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Tiantia Li
# I coorperated with the following classmates: Tianqi Bao
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt


class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        self.trainingData=xFeat
        self.label=y
        return self
    


    def predict(self, xFeat):
        

        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        #Calculate the Euclidean distance between two points 
        def EucliDis(pointx,pointy):
            dis=0;
            rowX=np.array(pointx)
            # d: number of features two points have
            d=len(rowX)
            rowY=np.array(pointy)
            for x in range(d-1):
                dis+=(rowX[x]-rowY[x])*(rowX[x]-rowY[x])
            dis=dis**0.5
            return dis
        
        yHat3 = [] # variable to store the estimated class label
        yHat5 = [] # variable to store the estimated class label
        yHat7 = [] # variable to store the estimated class label
        yHat9 = [] # variable to store the estimated class label
        yHat15 = []
        yHat=[yHat3,yHat5,yHat7,yHat9,yHat15]
        
        
        for i in range(len(xFeat.index)):
        #create a distance list for each testing data i, each entry j of the array represents the euclidean distance
        #betweeb the test data i and training data j
            distance=[]
            for j in range(len(self.trainingData.index)):
                #append a list to the distance list, the list we are appending each time consists of 1. the distance
                #and 2. the label of the training data j
                distance.append((EucliDis(xFeat.iloc[i],self.trainingData.iloc[j]),self.label.iloc[j]))
            #sort the data according to the first column of the distance list
            distance.sort(key=lambda x:x[0])
            #Since labels are binary, we can get the more popular label by adding up the label value
            #then compare with number of element divided by two
            count=0
            for r in range(len(self.k)):
                for c in range(self.k[r]):
                    count+=distance[c][1]
                if count>self.k[r]/2:
                    yHat[r].append(1)
                else: 
                    yHat[r].append(0)
        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    accLis = []
    
    for i in range(len(yHat)):
        acc=0
        for j in range(len(yHat[i])):
            if(yHat[i][j]==yTrue[j]):
                acc+=1
        accLis.append(acc/len(yHat[i]))
    return accLis


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    """
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    """
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn([4,7,9,15,31])
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    sns.lineplot(x=[4,7,9,15,31],y=trainAcc)
    
    print("Test Acc:", testAcc)
    sns.lineplot(x=[4,7,9,15,31],y=testAcc)
    


if __name__ == "__main__":
    main()