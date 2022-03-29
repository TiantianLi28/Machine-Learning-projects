''' THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS.
Tiantian Li '''
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt



def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column
    
    Revalent features to extract
    ----------------------------
    time_of_day: whether the data is obtained in the morning, afternoon, evening or midnight
    use dummy columns 
    drop midnight column
    Midnight: 0-5:59 am
    Morning: 6-11:59 am
    Afternoon: 12-17:59 pm
    Evening: 18-11:59 pm

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    #initialize columns and transfrom the date column from string to timestamp
    df['date']=pd.to_datetime(df['date'])
    df['time_of_day']='0'
    df['morning']=0
    df['afternoon']=0
    df['evening']=0
    df['midnight']=0
    
    #for each row in df, assign its time_of_day value.
    time_of_day=['midnight','morning','afternoon','evening']
    for row in range(len(df)):
        df.at[row,'time_of_day']=time_of_day[int(df.at[row,'date'].hour/6)]
        df.at[row,df.at[row,'time_of_day']]=1
        
    #drop redundant columns and the original date column  
    df=df.drop(columns=['midnight'])
    df = df.drop(columns=['date','time_of_day'])
    
    return df

def corrMatrix(df,y):
    '''
    Calculate correlation matrix and plot heatmap accordingly
    Variable
    --------
    df: pandas dataframe with extracted new columns
    y: pandas dataframe 
       Target variable
    
    '''
    df['y']=y
    corPlot=sns.heatmap(df.corr(method='pearson'),xticklabels=1,yticklabels=1)
    plt.show()
    return corPlot
    
    

    
def select_features(df,yTrain):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    #Rule of feature selection:
        #we want to keep features that are not very correlated with each other to reduce redundancy and avoid over-
        #complicating our model.
    df['y']=yTrain
    df=df.drop(columns=(['RH_1','T2','RH_2','RH_3','RH_7','RH_8','RH_9','Tdewpoint','T7','T8','T9','RH_6','T_out','T1','T5','T3','RH_out']))
    corPlot=sns.heatmap(df.corr(method='pearson'),annot=True,xticklabels=1,yticklabels=1)
    plt.show()
    df=df.drop(columns=('y'))
    return df,corPlot

def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data
    Use min max scaler to preprocess the data so that each feature of the data is within comparable range with the other features.
    Standard_scaling:
    ----------------
    Preprocess the data to have minimum value of 0 and maximum
    value of 1. The same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    min_max_scaler = preprocessing.MinMaxScaler()
    xTrainModified = pd.DataFrame(min_max_scaler.fit_transform(trainDF))
    xTestModified = pd.DataFrame(min_max_scaler.transform(testDF))
    return xTrainModified, xTestModified


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    parser.add_argument("--testYFile",
                        default="eng_yTest.csv",
                        help="filename of the test data")
    parser.add_argument("--trainYFile",
                        default="eng_yTrain.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    yTrain=pd.read_csv(args.trainYFile)
    yTest=pd.read_csv(args.testYFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    #plot correlation plot
    corPlot=corrMatrix(xNewTrain, yTrain)
    '''corPlotFig=corPlot.get_figure()
    corPlotFig.savefig('/Users/shirley/Desktop/hw3_template/corPlotTrain.png')'''
    corPlotTest=corrMatrix(xNewTest,yTest)
    '''corPlotFig=corPlotTest.get_figure()
    corPlotFig.savefig('/Users/shirley/Desktop/hw3_template/corPlotTest.png')'''
    # select the features
    xNewTrain, newPlotTrain = select_features(xNewTrain,yTrain)
    '''corPlotFig=newPlotTrain.get_figure()
    corPlotFig.savefig('/Users/shirley/Desktop/hw3_template/newPlotTrain.png')'''
    xNewTest, newPlotTest = select_features(xNewTest,yTest)
    '''corPlotFig=newPlotTest.get_figure()
    corPlotFig.savefig('/Users/shirley/Desktop/hw3_template/newPlotTest.png')'''
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
