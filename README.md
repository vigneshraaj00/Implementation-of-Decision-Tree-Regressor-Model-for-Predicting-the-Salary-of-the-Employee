# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.
## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vignesh Raaj S
RegisterNumber:  212223230239

import pandas as pd
data=pd.read_csv('Salary.csv')

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

<img width="439" height="251" alt="image" src="https://github.com/user-attachments/assets/d3df5ff1-a3e8-453f-8916-0342c6c5cf06" />

<img width="556" height="249" alt="image" src="https://github.com/user-attachments/assets/3dd250be-cf4a-4c6b-976f-9bfc147264f0" />

<img width="208" height="100" alt="image" src="https://github.com/user-attachments/assets/7a132f19-b891-4832-bc0c-5ec6b4031c73" />

<img width="313" height="258" alt="image" src="https://github.com/user-attachments/assets/638fb06d-0a00-46b0-a337-8e3fd3de4183" />

<img width="165" height="35" alt="image" src="https://github.com/user-attachments/assets/db5a8ac5-8cde-434f-9ce6-41b95be9fd25" />

<img width="242" height="43" alt="image" src="https://github.com/user-attachments/assets/a0243051-33da-4b18-9d41-122ce26f04cf" />

<img width="214" height="45" alt="image" src="https://github.com/user-attachments/assets/c0e8b90f-b1fa-408a-9d49-394b2017032c" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
