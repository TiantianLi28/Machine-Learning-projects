''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
   
    # load the train and test data assumes you'll use numpy
    xTrainBinary = pd.read_csv("xTrainBinary.csv")
    xTrainCount = pd.read_csv("xTrainCount.csv")
    yTrain = pd.read_csv("yTrain.csv")
    xTestBinary = pd.read_csv("xTestBinary.csv")
    xTestCount=pd.read_csv("xTestCount.csv")
    yTest = pd.read_csv("yTest.csv")
    
    
    bnb=BernoulliNB()
    yHat=bnb.fit(xTrainBinary, yTrain.values.ravel()).predict(xTestBinary)
    error=0
    print (yHat)
    for row in range(len(yHat)):
        error+=(yHat[row]-yTest.iloc[row,0])**2
    print('Binary:',error)
    
    mnb=MultinomialNB()
    yHat=mnb.fit(xTrainCount, yTrain.values.ravel()).predict(xTestCount)
    error=0
    print (yHat)
    for row in range(len(yHat)):
        error+=(yHat[row]-yTest.iloc[row,0])**2
    print('Count:',error)
    
    logReg = LogisticRegression()
    yHat=logReg.fit(xTrainBinary,yTrain.values.ravel()).predict(xTestBinary)
    error=0
    print (yHat)
    for row in range(len(yHat)):
        error+=(yHat[row]-yTest.iloc[row,0])**2
    print('Binary:',error)
    
    logReg = LogisticRegression(max_iter=400)
    yHat=logReg.fit(xTrainCount,yTrain.values.ravel()).predict(xTestCount)
    error=0
    print (yHat)
    for row in range(len(yHat)):
        error+=(yHat[row]-yTest.iloc[row,0])**2
    print('Count:',error)
    
    
    

if __name__ == "__main__":
    main()