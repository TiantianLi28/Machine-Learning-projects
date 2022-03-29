''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
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
        #initialize beta
        self.beta=np.ones((np.shape(xTrainOr)[1]+1,1))
        ones=np.ones((np.shape(xTrainOr)[0],1))
        xTrain=np.hstack((ones,xTrainOr))
        ones=np.ones((np.shape(xTestOr)[0],1))
        xTest=np.hstack((ones,xTestOr))
        trainStat=[]
        testStat=[]
        timeElap=[]
        timeS=time.time()
        for epoch in range(self.mEpoch):
            #shuffle data for each epoch
            combined=np.append(xTrain, yTrain, axis=1)
            np.random.shuffle(combined)
            xTrain=np.delete(combined, np.shape(combined)[1]-1, 1)
            yTrain=np.delete(combined, slice(np.shape(combined)[1]-1), 1)
            numBatch=int(np.shape(xTrain)[0]/self.bs)
            
            for b in range(numBatch):

                batchX=xTrain[b*self.bs:(b+1)*self.bs,:]
                batchY=yTrain[b*self.bs:(b+1)*self.bs,:]
                batchXT=np.transpose(batchX)
                batchYHat=np.matmul(batchX, self.beta)
                gradient=np.matmul(batchXT,(batchY-batchYHat))
                self.beta=self.beta+self.lr*gradient/self.bs   
            yTrainHat=self.predict(xTrain)
            yTestHat=self.predict(xTest)
            TrainMSETranspose=np.transpose(yTrain-yTrainHat)
            TrainMSE=float(np.matmul(TrainMSETranspose,(yTrain-yTrainHat))/np.shape(xTrainOr)[0])
            TestMSET=np.transpose(yTest-yTestHat)
            TestMSE=float(np.matmul(TestMSET,(yTest-yTestHat))/np.shape(xTestOr)[0])
            timeE=time.time()
            timeElapsed=timeE-timeS
            timeElap.append(timeElapsed)
            trainStat.append(TrainMSE)
            testStat.append(TestMSE)
                
        return trainStat,testStat,timeElap
    
    def plot_train_predict(self, xTrainOr, yTrain):
        #initialize beta and xTrain
        self.beta=np.ones((np.shape(xTrainOr)[1]+1,1))
        bs=30
        print (bs)
        ones=np.ones((np.shape(xTrainOr)[0],1))
        xTrain=np.hstack((ones,xTrainOr))
        #initialize a pd dataframe to store data for ploting
        #column_names=['0.6','0.55','0.5','0.45','0.4','0.3','0.2','0.1','0.05','0.01']
        column_names=['0.15','0.1','0.07','0.06','0.05','0.01']
        #column_names=['0.15','0.1','0.05','0.01']
        #Train and predict using epoch 5 and batch size 1
        #learningRate=[0.6,0.55,0.5,0.45,0.4,0.3,0.2,0.1,0.05,0.01]
        learningRate=[0.15,0.1,0.07,0.06,0.05,0.01]
        #learningRate=[0.1,0.01,0.001,0.0001]
        plot_data=[]
        for lrn in range(len(learningRate)):
            
            self.beta=np.ones((np.shape(xTrainOr)[1]+1,1))
            lr=learningRate[lrn]
            data=[]
            for epoch in range(9):
                combined=np.append(xTrain, yTrain, axis=1)
                np.random.shuffle(combined)
                xTrain=np.delete(combined, np.shape(combined)[1]-1, 1)
                yTrain=np.delete(combined, slice(np.shape(combined)[1]-1), 1)
                numBatch=int(np.shape(xTrain)[0]/bs)
                print(numBatch)
                for b in range(numBatch):
                    
                    batchX=xTrain[b*bs:(b+1)*bs,:]
                    batchY=yTrain[b*bs:(b+1)*bs,:]
                    batchXT=np.transpose(batchX)
                    batchYHat=np.matmul(batchX, self.beta)
                    gradient=np.matmul(batchXT,(batchY-batchYHat))
                    self.beta=self.beta+lr*gradient/bs
                    
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
    column_names=['1','10','30','559','1667','16670']
    bs=[1,10,30,559,1667,16670]
    lr=[0.01,0.05,0.07,0.55,0.6,0.55]
    train=[]
    test=[]
    timeE=[]
    for i in range(6):
        np.random.seed(args.seed) 
        model = SgdLR(lr[i], bs[i], 6)
    #model.plot_train_predict(xTrain, yTrain)
        trainStat, testStat,timeElap= model.train_predict(xTrain, yTrain, xTest, yTest)
        train+=trainStat
        test+=testStat
        timeE+=timeElap
    
    #print(trainStats)
    xTrain=np.array(train)
    xTrain=np.transpose(xTrain)
    
    xTest=np.array(test)
    xTest=np.transpose(xTest)
    
    xTime=np.array(timeE)
    xTime=np.transpose(xTime)
    df=pd.DataFrame()
    #df['MSE']=xTrain
    df['MSE']=xTest
    df['time']=timeE
    df['bs']=['1','1','1','1','1','1','10','10','10','10','10','10','30','30','30','30','30','30','559','559','559','559','559','559','1667','1667','1667','1667','1667','1667','16670','16670','16670','16670','16670','16670'
              ]
    df.loc[len(df)]=[0.26683152256247206,0.012369871139526367,'closed form']
    df.loc[len(df)]=[0.26683152256247206,0.013,'closed form']
    
    sns.lineplot(data=df,y='MSE',x='time',hue='bs')
    plt.show()

if __name__ == "__main__":
    main()

