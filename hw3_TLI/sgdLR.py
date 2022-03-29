''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
import numpy as np
import pandas as pd
import time
from lr import LinearRegression, file_to_numpy
import matplotlib.pyplot as plt


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrainOr, yTrain, xTestOr, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        #initialize beta
        self.beta=np.ones((np.shape(xTrainOr)[1]+1,1))
        ones=np.ones((np.shape(xTrainOr)[0],1))
        xTrain=np.hstack((ones,xTrainOr))
        ones=np.ones((np.shape(xTestOr)[0],1))
        xTest=np.hstack((ones,xTestOr))
        trainData=[]
        testData=[]
        timestart=time.time()
        for epoch in range(self.mEpoch):
            #shuffle data 
            combined=np.append(xTrain, yTrain, axis=1)
            np.random.shuffle(combined)
            xTrain=np.delete(combined, np.shape(combined)[1]-1, 1)
            yTrain=np.delete(combined, slice(np.shape(combined)[1]-1), 1)
            numBatch=int(np.shape(xTrain)[0]/self.bs)
            #update beta for each batch
            for b in range(numBatch):
                #fetch the target batch
                batchX=xTrain[b*self.bs:(b+1)*self.bs,:]
                batchY=yTrain[b*self.bs:(b+1)*self.bs,:]
                batchXT=np.transpose(batchX)
                batchYHat=np.matmul(batchX, self.beta)
                #calculate gradient
                gradient=np.matmul(batchXT,(batchY-batchYHat))
                self.beta=self.beta+self.lr*gradient/self.bs   
                
                #predict 
                yTrainHat=self.predict(xTrain)
                yTestHat=self.predict(xTest)
                #calculate MSE
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
                trainStats[(epoch+1)*b]=stat
            trainData.append(TrainMSE)
            testData.append(TestMSE)
        plotData=[trainData,testData]
        plotData=np.array(plotData)
        plotData=np.transpose(plotData)
        columnNames=['train MSE','test MSE']
        df=pd.DataFrame(plotData, columns=columnNames)
        plot=df.plot.line()
        plt.show()
        return trainStats
    
    def plot_train_predict(self, train, y):
        #shuffle the dataset and extract 40 percent
        combined=np.append(train, y, axis=1)
        np.random.shuffle(combined)
        newTrain=np.delete(combined, np.shape(combined)[1]-1, 1)
        newY=np.delete(combined, slice(np.shape(combined)[1]-1), 1)
        part_to_delete=int(0.4*np.shape(newTrain)[0])
        xTrainOr=np.delete(newTrain,slice(part_to_delete,np.shape(newTrain)[0]),axis=0)
        yTrain=np.delete(newY,slice(part_to_delete,np.shape(newY)[0]),axis=0)
        #initialize beta and xTrain
        self.beta=np.ones((np.shape(xTrainOr)[1]+1,1))
        ones=np.ones((np.shape(xTrainOr)[0],1))
        xTrain=np.hstack((ones,xTrainOr))
        #initialize a pd dataframe to store data for ploting
        column_names=['0.15','0.1','0.05','0.01']
        #Train and predict using epoch 5 and batch size 1
        learningRate=[0.15,0.1,0.05,0.01]
        plot_data=[]
        for lrn in range(len(learningRate)):
            
            self.beta=np.ones((np.shape(xTrainOr)[1]+1,1))
            lr=learningRate[lrn]
            data=[]
            for epoch in range(8):
                combined=np.append(xTrain, yTrain, axis=1)
                np.random.shuffle(combined)
                xTrain=np.delete(combined, np.shape(combined)[1]-1, 1)
                yTrain=np.delete(combined, slice(np.shape(combined)[1]-1), 1)
                numBatch=int(np.shape(xTrain)[0])
                for b in range(numBatch):
                    
                    batchX=xTrain[b:(b+1),:]
                    batchY=yTrain[b:(b+1),:]
                    batchXT=np.transpose(batchX)
                    batchYHat=np.matmul(batchX, self.beta)
                    gradient=np.matmul(batchXT,(batchY-batchYHat))
                    self.beta=self.beta+lr*gradient 
                    
                yTrainHat=self.predict(xTrain)
                TrainMSETranspose=np.transpose(yTrain-yTrainHat)
                TrainMSE=float(np.matmul(TrainMSETranspose,(yTrain-yTrainHat))/np.shape(xTrain)[0])
                data.append(TrainMSE)
            print(data)
            plot_data.append(data)
        plot_dataNew=np.array(plot_data)
        plotData=np.transpose(plot_dataNew)
            
        df=pd.DataFrame(plotData,columns=column_names)
        plot=df.plot.line()
        plt.show()
                
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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed) 
    model = SgdLR(args.lr, args.bs, args.epoch)
    #model.plot_train_predict(xTrain, yTrain)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

