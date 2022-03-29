from sklearn import preprocessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt

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

def logistic_Reg(xTrain,yTrain,xTest,yTest, xTrain_pca, xTest_pca):
    logReg = LogisticRegression(penalty='none')
    yHat=logReg.fit(xTrain,yTrain.values.ravel()).predict(xTest)
    y_pred_proba = logReg.predict_proba(xTest)[::,1]
    fpr, tpr, _ = metrics.roc_curve(yTest, y_pred_proba)
    plt.plot(fpr,tpr,label='normalized')
    yHat=logReg.fit(xTrain_pca,yTrain.values.ravel()).predict(xTest_pca)
    y_pred_proba = logReg.predict_proba(xTest_pca)[::,1]
    fpr, tpr, _ = metrics.roc_curve(yTest, y_pred_proba)
    plt.plot(fpr,tpr,label='pca')
    plt.legend()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def pca(xTrain,xTest):
    sk_pca = PCA(n_components=9)
    xTrain_pca=sk_pca.fit_transform(xTrain)
    xTest_pca=sk_pca.transform(xTest)
    return xTrain_pca, xTest_pca

def main():
    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv('q4xTrain.csv')
    yTrain = pd.read_csv('q4yTrain.csv')
    xTest = pd.read_csv('q4xTest.csv')
    yTest = pd.read_csv('q4yTest.csv')
    normalized_xTrain,normalized_xTest=normalize(xTrain,xTest)
    xTrain_pca,xTest_pca=pca(normalized_xTrain,normalized_xTest)
    prob=logistic_Reg(xTrain_pca, yTrain,xTest_pca,yTest)
    print(prob)
    logistic_Reg(normalized_xTrain,yTrain,normalized_xTest,yTest,xTrain_pca,xTest_pca)

if __name__ == "__main__":
    main()