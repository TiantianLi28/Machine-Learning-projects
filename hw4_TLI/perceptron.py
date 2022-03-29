''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import numpy as np
import pandas as pd
import time

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
        self.w=[0]*(len(xFeat.columns)+1)
        for epoch in range(self.mEpoch):
            #print('epoch:',epoch)
            for row in range(len(xFeat)): 
                predicted=self.predict(xFeat.iloc[[row]])
                yTrue=y.iloc[row,0]
                error=yTrue-predicted
                self.updateW(error,row,xFeat)
                #for i in range(len(xFeat.columns)):
                    #self.w[i+1]+=error*xFeat.iloc[row,i]
            yHat=self.predict(xFeat)
            error_count=calc_mistakes(yHat,y)
            stats[epoch]=error_count
            # if no mistake in predicting training data, stop training
            if error_count==0: 
                break
        most_pos=[]
        most_neg=[]
        sort_list=self.w
        most_pos = sorted(range(len(sort_list)), key = lambda sub: sort_list[sub])[-15:]
        most_neg = sorted(range(len(sort_list)), key = lambda sub: sort_list[sub])[:15]
  
       
        
        return stats, most_pos,most_neg
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
    '''
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
                parser.add_argument("xTrain",
                                    default="xTrainBinary.csv",
                                    help="filename for features of the training data")
                parser.add_argument("yTrain",
                                    default="yTrain.csv",
                                    help="filename for labels associated with training data")
                parser.add_argument("xTest",
                                    default="xTestBinary.csv",
                                    help="filename for features of the test data")
                parser.add_argument("yTest",
                                    default="xTrainBinary.csv",
                                    help="filename for labels associated with the test data")
                parser.add_argument("epoch", type=int, help="max number of epochs")
                parser.add_argument("--seed", default=334, 
                                    type=int, help="default seed number")
                
                args = parser.parse_args()
    '''
    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv("xTrainBinary.csv")
    xTest = pd.read_csv("xTestBinary.csv")
    xTrainc=pd.read_csv("xTrainCount.csv")
    xTestc=pd.read_csv("xTestCount.csv")
    yTrain = pd.read_csv("yTrain.csv")
    yTest = pd.read_csv("yTest.csv")
    wordMap=pd.read_csv('wordMap.csv')

    np.random.seed(334)
    model=Perceptron(1)
    trainStats,most_pos,most_neg = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset trained with binary")
    numError=calc_mistakes(yHat, yTest)
    print(numError)
    bpwords=[]
    for i in most_pos:
        bpwords.append(wordMap.iloc[i-1,0])
    print('Binary most_pos:', bpwords)
    bnwords=[]
    for i in most_neg:
        bnwords.append(wordMap.iloc[i-1,0])
    print('Binary most_neg:', bnwords)

    model=Perceptron(50)
    trainStats,most_pos,most_neg = model.train(xTrainc, yTrain)
    print(trainStats)
    yHat = model.predict(xTestc)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset trained with count")
    numError=calc_mistakes(yHat, yTest)
    print(numError)
    cpwords=[]
    for i in most_pos:
        cpwords.append(wordMap.iloc[i-1,0])
    print('Count most_pos:', cpwords)
    cnwords=[]
    for i in most_neg:
        cnwords.append(wordMap.iloc[i-1,0])

    print('Count most_neg:', cnwords)

if __name__ == "__main__":
    main()