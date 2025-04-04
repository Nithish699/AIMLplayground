#==============Importing libraries====================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#===============Data collecting====================
car_dataset=pd.read_csv("car data.csv")
print(car_dataset.head())
print("shape:", car_dataset.shape)
print("Description/Statistical",car_dataset.describe())
print("Information about dataset",car_dataset.info())
print("Checking Null :",car_dataset.isnull().sum())

#=============counting unique values================

print("Counting unique values in Fuel type",car_dataset.Fuel_Type.value_counts())
print("Counting unique values in Seller Type",car_dataset.Seller_Type.value_counts())
print("Counting unique values in Transmission",car_dataset.Transmission.value_counts())

#==============Encoding the categorical data==============

car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

print(car_dataset.head(10))
print(car_dataset.columns)


# ==================Splitting the data and Target==============

X=car_dataset.drop(['Car_Name','Selling_Price'],axis=1)#input
Y=car_dataset['Selling_Price']#output

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=2)

#================Model Training========================

model=LinearRegression()
model.fit(x_train,y_train)

#==================Model Evalution=========
#predicting on Traing data
training_data_prediction=model.predict(x_train)

#============R sqaure Error====================
error_score=metrics.r2_score(y_train,training_data_prediction)
print("R2 score: ",error_score)


#================Visualize==================
plt.scatter(y_train,training_data_prediction)
plt.xlabel("ActualPrice")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs  Predicted Prices")
plt.show()

#=========Predicting on Test Data=====

test_data_pred=model.predict(x_test)
error_score=metrics.r2_score(y_test,test_data_pred)
print("Error:",error_score)



#=============Scaling the data Lasso Regresssion=======

#=====================save the model=============
import joblib

# Save the model
joblib.dump(model, 'car_price_prediction_model.pkl')
print("Model saved!")


