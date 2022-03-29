''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold 

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}

        xFeat=pd.DataFrame(xFeat)
        kf = KFold(n_splits=5)
        total_error=0
        for train_index, test_index in kf.split(xFeat):
            xTrain, xTest = xFeat.iloc[train_index], xFeat.iloc[test_index]
            yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

            self.w=[0]*(len(xTrain.columns)+1)
            for epoch in range(self.mEpoch):
                #print('epoch:',epoch)
                for row in range(len(xTrain)): 
                    predicted=self.predict(xTrain.iloc[[row]])
                    yTrue=yTrain.iloc[row,0]
                    error=yTrue-predicted
                    self.updateW(error,row,xTrain)
                    #for i in range(len(xFeat.columns)):
                        #self.w[i+1]+=error*xFeat.iloc[row,i]
                yHat=self.predict(xTrain)
                error_count=calc_mistakes(yHat,yTrain)
                # if no mistake in predicting training data, stop training
                if error_count==0: 
                    break
            testHat=self.predict(xTest)
            print('1',total_error)
            total_error+=calc_mistakes(testHat,yTest)
        print('total Error:', total_error)
        return total_error

    """
    update w depending on the error and the row
    if no error, do nothing
    if there is error (1 or -1):
        multiply the row by error;
        add a row to xfeat[row] to update intercept(w0);
        add w with row
        update in self.w
    """
    def updateW(self,error,rowNum,xFeat):
        if error==0:
            return
        else:
            #wi:1*#of_feature+1
            wi=np.array(self.w)
            
            row=xFeat.iloc[[rowNum]].to_numpy()
            #print(1,row)
            row=error*row
            #print(2,row)
            row=np.insert(row,0,error)
            #print(3,row)
            
            self.w=np.add(wi,row)

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.
        To optimize running time, use matrix multiplication.
        xFeat: n rows by f features
        w: 

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """

        yHat = []
        #xFeat: 1*2443
        xFeat=xFeat.to_numpy()
        xFeat=np.insert(xFeat,0,1,1)
        xFeat=np.transpose(xFeat)
        w=np.array(self.w)
        w=w.reshape(1,len(w))
        #print(w.shape,xFeat.shape)
        result=np.matmul(w,xFeat)
        result=np.transpose(result)
        for row in result:
            if row >=0:
                yHat.append(1)
            else:
                yHat.append(0)

        '''for row in range(len(xFeat)):
            predictValue=self.w[0]
            for entry in range(len(xFeat.columns)):
                predictValue+=xFeat.iloc[row,entry]*self.w[entry+1]
            if predictValue>=0:
                yHat.append(1)
            else: 
                yHat.append(0)'''
        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    error_count=0
    for i in range(len(yHat)):
        error_count+=(yHat[i]-yTrue.iloc[i,0])**2
    return error_count


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    xTrain = pd.read_csv("xTrainBinary.csv")
    xTest = pd.read_csv("xTestBinary.csv")
    xTrainc=pd.read_csv("xTrainCount.csv")
    xTestc=pd.read_csv("xTestCount.csv")
    yTrain = pd.read_csv("yTrain.csv")
    yTest = pd.read_csv("yTest.csv")
    best_binary_epoch=0
    best_binary_error=10000
    best_count_epoch=0
    best_count_error=10000
    binary_epoch=[]
    count_epoch=[]
    #test_epoch=[5,10,20,30,40,50,55]
    test_epoch=[5,6,7,10,20,35,50]
    for epoch in test_epoch:
        binary_error=0
        count_error=0
        np.random.seed(334)

        print('epoch:',epoch)
        model=Perceptron(epoch)
        binary_error=model.train(xTrain, yTrain)

        
        print("Average mistakes on the test dataset trained with binary:",binary_error)
        if binary_error<best_binary_error:
            best_binary_epoch=epoch
            best_binary_error=binary_error
        binary_epoch.append([epoch,binary_error])


        model=Perceptron(epoch)
        count_error=model.train(xTrainc, yTrain)

        print("Average mistakes on the test dataset trained with count:",count_error)
        if count_error<best_count_error:
            best_count_epoch=epoch
            best_count_error=count_error
        count_epoch.append([epoch,count_error])

    print('binary epoch:')
    print(binary_epoch)
    print('count epoch:')
    print(count_epoch)
    print('best binary epoch:',best_binary_epoch,'best epoch error:',best_binary_error)
    print('best count epoch:',best_count_epoch,'best epoch error:',best_count_error)
if __name__ == "__main__":
    main()