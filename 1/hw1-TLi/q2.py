# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Tiantia Li
# I coorperated with the following classmates: Tianqi Bao
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
from sklearn import datasets

#load iris dataset from sklearn
iris=datasets.load_iris()

#transform to a pandas dataframe
df=pd.DataFrame(iris.data,
                  columns = iris.feature_names)

#target contains the type of iris encoded in 0,1,and 2 corresponding to the index of three types of iris stored in taget_name
typeNumber=iris.target
typeName=iris.target_names
#create a new column with iris type
dfWithNum=df.assign(typenum=typeNumber)
dfWithType=dfWithNum.assign(types=typeName[typeNumber])


#boxplot of distribution of petal length 
bp_petalLength=dfWithType.boxplot(column=['petal length (cm)'],by=['types'])
#boxplot of distribution of petal widtch 
bp_petalWidth=dfWithType.boxplot(column=['petal width (cm)'],by=['types'])
#boxplot of distribution of sepal length 
bp_sepalLength=dfWithType.boxplot(column=['sepal length (cm)'],by=['types'])
#boxplot of distribution of sepal width
bp_sepalWidth=dfWithType.boxplot(column=['sepal width (cm)'],by=['types'])

#create a colormap for scatterplot
colors={'setosa':'pink','versicolor':'orange','virginica':'yellow'}
#plot scatterplot
sns.lmplot(x='petal length (cm)', y='petal width (cm)', data=dfWithType, hue='types', fit_reg=False)
sns.lmplot(x='sepal length (cm)', y='sepal width (cm)', data=dfWithType, hue='types', fit_reg=False)
