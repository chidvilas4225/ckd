import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import keras as k



dataframe=pd.read_csv('C:\\Users\\Suri\\Documents\\chronic kidney disease\\kidney_disease.csv')

columns_to_retain = ["sg","al","sc","hemo","pcv","htn","classification"]


dataframe=dataframe.drop([col for col in dataframe.columns if not col in columns_to_retain],axis=1)

dataframe=dataframe.dropna(axis=0)

for column in dataframe.columns:
    if dataframe[column].dtype==np.number:
        continue
    dataframe[column]=LabelEncoder().fit_transform(dataframe[column])  



X= dataframe.drop(['classification'],axis=1)
y=dataframe['classification']



x_scaler=MinMaxScaler()
x_scaler.fit(X)
column_names=X.columns
X[column_names]=x_scaler.transform(X)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
Y_pred=classifier.predict(X_test)

Y_pred



from sklearn import metrics
print("Accuracy Score: with linear kernel")

print(metrics.accuracy_score(y_test,Y_pred))

from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
classifier.fit(X_train,y_train)
Y_pred=classifier.predict(X_test)

print("Accuracy Score: with dealut rbf kernel")

print(metrics.accuracy_score(y_test,Y_pred))