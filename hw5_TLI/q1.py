''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
from sklearn import preprocessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def normalize(trainDF,testDF):
    standard_scaler = preprocessing.StandardScaler()
    xTrainModified = pd.DataFrame(standard_scaler.fit_transform(trainDF))
    xTestModified = pd.DataFrame(standard_scaler.transform(testDF))
    return xTrainModified, xTestModified  
'''
def normalize(trainDF,testDF):
    xTrainModified = pd.DataFrame(preprocessing.normalize(trainDF))
    xTestModified = pd.DataFrame(preprocessing.normalize(testDF))
    return xTrainModified, xTestModified  
'''
'''
def logistic_Reg(xTrain,yTrain,xTest,yTest):
    logReg = LogisticRegression(penalty='none')
    yHat=logReg.fit(xTrain,yTrain.values.ravel()).predict(xTest)
    count=0
    for row in range(len(yHat)):
        if yHat[row]==yTest.iloc[row,0]:
            count+=1
    return count/len(yHat)
'''
def logistic_Reg(xTrain,yTrain,xTest,yTest):
    logReg = LogisticRegression(penalty='none')
    yHat=logReg.fit(xTrain,yTrain.values.ravel()).predict(xTest)
    return accuracy_score(yTest,yHat)

def pca(xTrain,xTrainCol):
    n_com=[2,3,4,5,6,7,8,9]
    variance=[]
    for cpn in n_com:
        sk_pca = PCA(n_components=cpn)
        xTrain_pca=sk_pca.fit_transform(xTrain)
        count=0
        for var in sk_pca.explained_variance_ratio_:
            count+=var
        variance.append(count)
    print(variance)
    col=xTrainCol.columns
    comp=pd.DataFrame(sk_pca.components_, columns=col)
    print("components")
    print(comp)
    return xTrain_pca

def pca_transform(xTrain,xTest):
    sk_pca = PCA(n_components=9)
    xTrain_pca=sk_pca.fit_transform(xTrain)
    xTest_pca=sk_pca.transform(xTest)
    return xTrain_pca,xTest_pca

def main():
    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv('q4xTrain.csv')
    yTrain = pd.read_csv('q4yTrain.csv')
    xTest = pd.read_csv('q4xTest.csv')
    yTest = pd.read_csv('q4yTest.csv')
    
    normalized_xTrain,normalized_xTest=normalize(xTrain,xTest)
    prob=logistic_Reg(normalized_xTrain, yTrain,normalized_xTest,yTest)
    print(prob)
    pca(normalized_xTrain,xTrain)
    xTrain_pca, xTest_pca=pca_transform(normalized_xTrain,normalized_xTest)
    prob=logistic_Reg(xTrain_pca, yTrain,xTest_pca,yTest)
    print(prob)
    

if __name__ == "__main__":
    main()