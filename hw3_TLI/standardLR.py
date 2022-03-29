''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
import numpy as np
import pandas as pd
import time
from numpy.linalg import inv

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrainOr, yTrain, xTestOr, yTest):
        """
        See definition in LinearRegression class
        """
        #training with closed form linear regression
        timestart=time.time()
        trainStats = {}
        #add dummy column to xTrain and xTest
        ones=np.ones((np.shape(xTrainOr)[0],1))
        xTrain=np.hstack((ones,xTrainOr))
        ones=np.ones((np.shape(xTestOr)[0],1))
        xTest=np.hstack((ones,xTestOr))
        
        #Train linear regression model with xTrain
        TrainT=np.transpose(xTrain)
        TrainTM=np.matmul(TrainT,xTrain)
        TrainTMI=inv(TrainTM)
        TrainTMIM=np.matmul(TrainTMI,TrainT)
        TrainTMIMY=np.matmul(TrainTMIM,yTrain)
        self.beta=TrainTMIMY
        
        #predict
        yTrainHat=self.predict(xTrain)
        yTestHat=self.predict(xTest)
        
        #Calculate MSE for predicting training and testing data
        
        TrainMSETranspose=np.transpose(yTrain-yTrainHat)
        TrainMSE=float(np.matmul(TrainMSETranspose,(yTrain-yTrainHat))/np.shape(xTrainOr)[0])
        TestMSET=np.transpose(yTest-yTestHat)
        TestMSE=float(np.matmul(TestMSET,(yTest-yTestHat))/np.shape(xTestOr)[0])
        
        TimeEnd=time.time()
        timeElapsed=TimeEnd-timestart
        stat={}
        stat['time']=timeElapsed
        stat['train-mse']=TrainMSE
        stat['test-mes']=TestMSE
        trainStats['0']=stat
        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
