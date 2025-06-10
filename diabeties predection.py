import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data=pd.read_csv(r"D:\diabetes (1).csv")
print(data.shape)
print(data.head())
print(data.info())
print(data.describe())

result=data['Outcome'].value_counts()
print(result)
# no need of operation on  imbalanced data
#feature extraction 
X=data.drop(columns='Outcome' , axis=1)
Y=data['Outcome']

print(X.head())
print(Y.head())

X_train , X_test , Y_train ,Y_test=train_test_split(X,Y , random_state=2 , test_size=0.2 , stratify=Y)

print(X_train.shape , X_test.shape)

scaler=StandardScaler()

Xtrainsta=scaler.fit_transform(X_train)
Xteststa=scaler.transform(X_test)


# fitting in a model , predicting and checking accuracy 

classifer=svm.SVC(kernel='linear')
classifer.fit(Xtrainsta , Y_train)

train_prediction=classifer.predict(Xtrainsta)
train_accuracy=accuracy_score(train_prediction , Y_train)

print("Accuracy of train data" ,train_accuracy)

test_prediction=classifer.predict(Xteststa)
test_accuracy=accuracy_score(test_prediction , Y_test)

print("accuracy of test data" , test_accuracy)
















































