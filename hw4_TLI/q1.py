''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    X=filename['email']
    y=filename['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return  X_train, X_test, y_train, y_test


def build_vocab_map(file):
    wordMap=dict()
    for i in file:
        checked=dict()
        str=i.split()
        for j in str:
            if checked.get(j,0)==0:
                wordMap[j]=wordMap.get(j,0)+1
                checked[j]=1
    for key in list(wordMap.keys()):
        if wordMap[key]<30:
            del wordMap[key]
    print(len(wordMap))
    return wordMap



def construct_binary(xTrain,xTest,wordMap):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    lis=[]
    key_list=list(wordMap.keys())
    #list containing words that appeared in at least 30 emails

    for i in xTrain:
        #list filled with 0's
        value_list=[0]*len(key_list)
        str=i.split()
        for j in range(len(key_list)):
            for k in str:
                if k==key_list[j]:
                    if value_list[j]!=1:
                        value_list[j]=1
        lis.append(value_list)
    lis=pd.DataFrame(lis,columns=key_list)
    print(lis)
    
    lisTest=[]
    for i in xTest:
        #list filled with 0's
        value_list=[0]*len(key_list)
        str=i.split()
        for j in range(len(key_list)):
            for k in str:
                if k==key_list[j]:
                    if value_list[j]!=1:
                        value_list[j]=1
        lisTest.append(value_list)
    lisTest=pd.DataFrame(lisTest,columns=key_list)
    return lis,lisTest


def construct_count(xTrain,xTest,wordMap):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """ 
    lis=[]
    key_list=list(wordMap.keys())
    #list containing words that appeared in at least 30 emails

    for i in xTrain:
        #list filled with 0's
        value_list=[0]*len(key_list)
        str=i.split()
        for j in range(len(key_list)):
            for k in str:
                if k==key_list[j]:
                        value_list[j]+=1
        lis.append(value_list)
    
    lis=pd.DataFrame(lis,columns=key_list)
    print(lis)
    
    lisTest=[]
    for i in xTest:
        #list filled with 0's
        value_list=[0]*len(key_list)
        str=i.split()
        for j in range(len(key_list)):
            for k in str:
                if k==key_list[j]:
                    value_list[j]+=1
        lisTest.append(value_list)
    lisTest=pd.DataFrame(lisTest,columns=key_list)
    return lis,lisTest

def construct_binary_whole(xFeat,wordMap):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    lis=[]
    key_list=list(wordMap.keys())
    #list containing words that appeared in at least 30 emails

    for i in xFeat:
        #list filled with 0's
        value_list=[0]*len(key_list)
        str=i.split()
        for j in range(len(key_list)):
            for k in str:
                if k==key_list[j]:
                    if value_list[j]!=1:
                        value_list[j]=1
        lis.append(value_list)
    lis=pd.DataFrame(lis,columns=key_list)
    return lis



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    
    #process input data
    dataFile=pd.read_csv(args.data)
    dataFile.columns=['emails']
    dataFile[['label','email']]=dataFile.emails.str.split(' ',1,expand=True)
    dataFile=dataFile.drop('emails',1)
    #
    #split test/train data
    xTrain,xTest,yTrain,yTest=model_assessment(dataFile)

    
    
    yTrain.to_csv("yTrain1.csv", index=False)
    yTest.to_csv("yTest1.csv", index=False)
    
    wordMap=build_vocab_map(xTrain)
    pd.DataFrame(list(wordMap.keys())).to_csv('wordMap.csv',index=False)
    xTrainBinary,xTestBinary=construct_binary(xTrain,xTest,wordMap)
    xTrainBinary.to_csv("xTrainBinary.cvs", index=False)
    xTestBinary.to_csv("xTestBinary.cvs", index=False)
    
    xTrainCount,xTestCount=construct_count(xTrain,xTest,wordMap)
    xTrainCount.to_csv("xTrainCount.csv", index=False)
    xTestCount.to_csv("xTestCount.csv", index=False)

if __name__ == "__main__":
    main()
