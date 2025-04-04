import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#import warnings
#warnings.simplefilter("ignore")

df=pd.read_csv("IRIS.csv")
print(df.head())
print("Describe DataFrame \n",df.describe())
print("Inoformation \n",df.info())
print("checking Null\n",df.isnull().sum())#NO null good to move
print("Size of dateset\n",df.shape)
print(f"Unique Species\n{len(df["species"])}",df["species"].value_counts())

#==================================Splitting the data====================

X=df.iloc[:,:4]# iloc[row_start:row_end,column_start:column_end]
Y=df.iloc[:,4]
print(X)
print(Y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=0)
print(x_train.shape,"is train size")
print(x_test.shape,"is test size")
#=============Training============

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)

#==================evaluating=====
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print("Accuracy score of model ",score)

#=======predicting with new value===========
predicted=model.predict([[6.4,3.2,4.5,1.5]])
print(predicted)


#===================== Save the model========================
import joblib
joblib.dump(model, 'iris_classification_model.pkl')
print("Model saved!")
