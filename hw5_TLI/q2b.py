''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
import seaborn as sns

class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    trees=None

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.maxFeat=maxFeat
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
    
    def maj(self, lis):
        count=0
        for i in lis:
            count+=i
        if count >= len(lis)/2:
            return 1
        else:
            return 0

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        xFeat=pd.DataFrame(xFeat)
        y=pd.DataFrame(y)
        report={}
        trees=[]
        prediction=defaultdict(list)
        c=list(xFeat.index.values)
        for b in range(self.nest):
            #xFeatSelcted with last column as y
            xFeat['y']=y
            #sampel with replacement
            xFeatSelected=xFeat.sample(n=len(xFeat), replace=True)
            #build tree
            dcTree=DecisionTreeClassifier(criterion=self.criterion, min_samples_leaf=self.minLeafSample, max_depth=self.maxDepth, max_features=self.maxFeat, random_state=0)
            dcTree.fit(xFeatSelected.iloc[:,:-1],xFeatSelected.iloc[:,-1:])
            trees.append(dcTree)
            a=list(xFeatSelected.index.values)
            c=list(xFeat.index.values)
            #find rows/samples not in training set
            #oob=pd.DataFrame(xFeatSelected.iloc[list(set(c)-set(a))])
            oob=list(set(c)-set(a))
            errorCount=0
            for r in oob:
                row=xFeat.loc[[r]]
                result=dcTree.predict(row.iloc[:,:-1])
                #print(result)
                prediction[r].append(result)
                #print(prediction[r])
            for p in prediction:
                row=xFeat.loc[[p]]
                result= self.maj(list(prediction[p]))
                if result!=row.iloc[0]['y']:
                    errorCount+=1
            report[len(trees)]=errorCount/len(prediction)
            print(errorCount, len(prediction))
            print(report)
            
        self.trees=trees
        return report

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
            Predicted response per sample
        """
        yHat = []
        for row in xFeat:
            Hat=0
            for tree in self.trees:
                Hat+=tree.predict([row])
            if Hat > len(self.trees)/2:
                yHat.append(1)
            else:
                yHat.append(0)
        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    
    xTrain = file_to_numpy("q4xTrain.csv")
    yTrain = file_to_numpy("q4yTrain.csv")
    xTest = file_to_numpy("q4xTest.csv")
    yTest = file_to_numpy("q4yTest.csv")

    max_depth=[3,5,7,9,11]
    min_leaf=[5,10,15,20,25]
    max_feat=[5,6,7,8,9]
    criterionList=['gini','entropy']

    #np.random.seed(args.seed)   
    np.random.seed(334) 
    minLeafResult=[]
    max_featResult=[]
    max_depthResult=[]
    criterionResult=[]
    for ml in criterionList:
        model = RandomForest(30,8, ml ,9, 10)
        trainStats = model.train(xTrain, yTrain)
        minLeafResult.append(list(trainStats.values()))
    resultData=pd.DataFrame(minLeafResult,columns=list(trainStats.keys()))
    resultData=resultData.T
    resultData.columns=criterionList
    print()
    sns.lineplot(data=resultData)
    yHat = model.predict(xTest)
    print(accuracy_score(yTest,yHat))

if __name__ == "__main__":
    main()