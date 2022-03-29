''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
from typing import NoReturn
import numpy as np 
import pandas as pd
import math
from sklearn.metrics import accuracy_score

class Node(object):
    left= None
    right=None
    #spliting value
    value=-1
    #spliting feature
    feature=-1
    #current depth
    depth=0
    label=-1

    def __init__(self,feature,value,depth,label):
        self.feature=feature
        self.value=value
        self.depth=depth
        self.label=label
        
class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    node=None
    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.node=None

    def train(self, xFeat, y):
        """
        Train the decision tree model.

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
        #create a new 'label' column in xFeat 
        xFeat['label']=y

        #check if the list is pure already
        def checkPure(Feat):
            label=Feat[0]['label']
            for i in range(len(Feat)):
                if Feat[i]['label'] is not label:
                    return False
                return True

        #calculate giniIndex
        #groups: a list consisting of two lists, each containing the labels column of a group after a split
        def giniIndex(left,right):
            #sizes of the left and right groups
            leftSize=float(len(left))
            rightSize=float(len(right))
            totalCount=float(leftSize+rightSize)
            
            countL=0.0
            for row in range(len(left)):
                countL+=left[row][-1]
            AProbL=float(countL/leftSize)
            BProbL=float(1-AProbL)
            leftGini=1.0-AProbL**2-BProbL**2
            countR=0.0
            
            for row in range(len(right)):
                countR+=right[row][-1]
            AProbR=float(countR/rightSize)
            BProbR=float(1-AProbR)
            rightGini=1.0-AProbR**2-BProbR**2
            #print("GINI:",(leftSize/totalCount)*leftGini+(rightSize/totalCount)*rightGini)
            return (leftSize/totalCount)*leftGini+(rightSize/totalCount)*rightGini

        #calculate informationGain
        #left,right: list type
        def entropy(left,right):
            leftSize=len(left)
            rightSize=len(right)
            totalCount=leftSize+rightSize
            labelA=left[0][-1]
            countA=0.0
            countB=0.0
            for row in range(len(left)):
                if(left[row][-1]==labelA):
                    countA+=1
                else:
                    countB+=1
            leftCount=countA+countB
            if(float(countA/leftCount)!=0 and float(countB/leftCount)!=0):
                leftEntropy=((countA/leftCount)*float(math.log(float(countA/leftCount),2))+(countB/leftCount)*float(math.log(float(countB/leftCount),2)))
                leftEntropy=0.0-leftEntropy
            else:
                leftEntropy=0.0
            countA=0.0
            countB=0.0
            labelA=right[0][-1]
            for row in range(len(right)):
                if(right[row][-1]==labelA):
                    countA+=1
                else:
                    countB+=1
            rightCount=countA+countB
            if(float(countA/leftCount)!=0 and float(countB/leftCount)!=0):
                rightEntropy=((countA/rightCount)*float(math.log(float(countA/rightCount),2))+(countB/rightCount)*float(math.log(float(countB/rightCount),2)))
                rightEntropy=0.0-rightEntropy
            else:
                rightEntropy=0.0
            return (leftCount/totalCount)*leftEntropy+(rightCount/totalCount)*rightEntropy

        #split according to given feature and values. 
        #return two lists: first consist of value smaller than the given value
        #second with values larger than or equal to given value
        def split(Feat,feat,value):
            left=[]
            right=[]
            for row in range(len(Feat)):
                if Feat.iloc[row,feat]<value:

                    left.append(Feat.iloc[row])
                else:
                    right.append(Feat.iloc[row])
            return left,right
        
        '''
        Find the best spliting point. 
        First sort the dataframe by column to avoid having leaf smaller than minLeafSample.
        Then go through each row and column to find the best spliting point. 
        return the best spliting value and feature.
        '''
        def best_split(Feat):
            bestScore=1000
            bestValue=-1
            bestFeat=-1
            for columnNum in range(Feat.shape[1]-1):
                column=Feat.columns[columnNum]
                #sorting
                xFeatSorted=Feat.sort_values(by=[column])
                feat=columnNum
                preValue=0
                #print("lenth:",len(xFeatSorted))
                #print(self.minLeafSample,(len(xFeatSorted)-self.minLeafSample))
                #print("up",self.minLeafSample,(len(xFeatSorted)-self.minLeafSample))
                for row in range (self.minLeafSample,(len(xFeatSorted)-self.minLeafSample)):
                    value=xFeatSorted.iloc[row,columnNum]
                    #print('value:',value)
                    if(preValue!=value):
                        #print('preValue:',preValue)
                        preValue=value
                        left=(xFeatSorted.iloc[:row]).values.tolist()
                        right=(xFeatSorted.iloc[(row+1):]).values.tolist()
                        if(self.criterion == 'gini'):
                            score=giniIndex(left,right)
                        if(self.criterion == 'entropy'):
                            score=entropy(left,right)
                        if(score<bestScore):
                            bestScore=score
                            bestValue=value
                            bestFeat=feat
            if bestFeat!=-1:
                print('bestCol:',bestFeat,'bestVal:',bestValue)
            return bestValue,bestFeat    
            
        '''
        return the majority label
        assuming label is in binary
        '''
        def majClass(arr):
            colNum=len(arr.columns)
            countA=0
            for row in range(len(arr)):
                labelA=arr.iloc[row,colNum-1]
                countA+=labelA
            if(countA>len(arr)/2):
                return 1.0
            else:
                return 0.0

        '''
        recursively grow on the root node while the depth is below maxDepth
        '''
        def growTree(LFeat, depth):
            Feat=pd.DataFrame(LFeat)
            if depth>self.maxDepth:
                print('entered')
                return Node(-1,-1,depth,majClass(Feat))
            bestValue,bestFeat=best_split(Feat)
            cur=Node(bestFeat,bestValue,depth,majClass(Feat))
            if(bestValue==-1):
                return cur
            xFeatLeft,xFeatRight=split(Feat,bestFeat,bestValue)
            if(depth<self.maxDepth+1):
                #print(len(xFeatLeft),len(xFeatRight))
                if(checkPure(xFeatLeft)!=True):
                    cur.left=growTree(xFeatLeft,depth+1)
                if(checkPure(xFeatRight)!=True):
                    cur.right=growTree(xFeatRight,depth+1)
            return cur
        
        xFeat_copy=xFeat.copy()
        bestValue,bestFeat=best_split(xFeat_copy)
        root= Node(bestFeat,bestValue,1,majClass(xFeat))
        xFeatLeft,xFeatRight=split(xFeat,bestFeat,bestValue)
        if(2<self.maxDepth+1):
            if(checkPure(xFeatLeft)!=True):
                root.left=growTree(xFeatLeft,2)
            if(checkPure(xFeatRight)!=True):
                root.right=growTree(xFeatRight,2)

        self.node=root


        return self
    

    def predict(self, xFeat):
        '''
        print out the tree
        Check if tree is building according to the maxDepth and minLeafSample given
        '''
        
        def traverse(currentNode):
            if(currentNode is None):
                return 
            else:
                traverse(currentNode.left)
                #print(currentNode.value)
                '''if(currentNode.value==-1):
                    print("hasLeftNode?:",currentNode.left is None)
                    print("depth:",currentNode.depth)
                    print('label',currentNode.label)'''
                traverse(currentNode.right)
            
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
        yHat = [] # variable to store the estimated class label
        # TODO
        def travelTree(curNode,curRow):
            if curNode.left is None:
                return curNode.label
            nodeFeat=curNode.feature
            nodeValue=curNode.value
            rowValue=xFeat.iloc[curRow,nodeFeat]
            if rowValue<nodeValue:
                return travelTree(curNode.left,curRow)
            else:
                return travelTree(curNode.right,curRow)
        #traverse(self.node)
        for row in range(len(xFeat)):
            #curRow=xFeat.iloc[row]
            predicted=travelTree(self.node,row)
            yHat.append(predicted)
        #traverse(self.node)
        print(yHat)
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    
    """
    
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    
    '''parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")'''
    
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
    minLeaf=[15,20,25,30,35]
    maxDepth=[2,3,4,5,6]
    testAccMaxDepth1=[]
    trainAccMaxDepth1=[]
    testAccMaxDepth2=[]
    trainAccMaxDepth2=[]
    testAccMinLeaf1=[]
    trainAccMinLeaf1=[]
    testAccMinLeaf2=[]
    trainAccMinLeaf2=[]
    # create an instance of the decision tree using gini
    # while minLeaf=50, how does accuracy vary for different maxDepth?
    for mls in minLeaf:
        md=5
        dt1 = DecisionTree('gini', md, mls)
        trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
        testAccMinLeaf1.append(testAcc1)
        trainAccMinLeaf1.append(trainAcc1)
        dt = DecisionTree('entropy',md, mls)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        testAccMinLeaf2.append(testAcc)
        trainAccMinLeaf2.append(trainAcc)
        print("GINI Criterion ---------------")
        print("Training Acc:", trainAcc1)
        print("Test Acc:", testAcc1)
        print("Entropy Criterion ---------------")
        print("Training Acc:", trainAcc)
        print("Test Acc:", testAcc)
    for md in maxDepth:
        mls=15
        dt1 = DecisionTree('gini', md, mls)
        trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
        testAccMaxDepth1.append(testAcc1)
        trainAccMaxDepth1.append(trainAcc1)
        dt = DecisionTree('entropy',md, mls)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        testAccMaxDepth2.append(testAcc)
        trainAccMaxDepth2.append(trainAcc)
        print("GINI Criterion ---------------")
        print("Training Acc:", trainAcc1)
        print("Test Acc:", testAcc1)
        print("Entropy Criterion ---------------")
        print("Training Acc:", trainAcc)
        print("Test Acc:", testAcc)
    minLeafResult=pd.DataFrame({'testAccuracyUsingGini':testAccMinLeaf1,"testAccuracyUsingEntropy":testAccMinLeaf2,"trainAccuracyUsingGini":trainAccMinLeaf1,"trainAccuracyUsingEntropy":trainAccMinLeaf2},index=minLeaf)
    maxDepthResult=pd.DataFrame({'testAccuracyUsingGini':testAccMaxDepth1,"testAccuracyUsingEntropy":testAccMaxDepth2,"trainAccuracyUsingGini":trainAccMaxDepth1,"trainAccuracyUsingEntropy":trainAccMaxDepth2},index=maxDepth)
    linePlot1=minLeafResult.plot.line()
    linePlot2=maxDepthResult.plot.line()
    


if __name__ == "__main__":
    main()
