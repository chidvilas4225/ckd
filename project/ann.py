import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import keras as k

df=pd.read_csv('C:\\Users\\Suri\\Documents\\chronic kidney disease\\kidney_disease.csv')
df.head()

columns_to_retain = ["sg","al","sc","hemo","pcv",'htn',"classification"]

#drop the columns that are not in columns_to_retain
df=df.drop([col for col in df.columns if not col in columns_to_retain],axis=1)
#drop the rows with na or missing values
df=df.dropna(axis=0)
df.head()

for column in df.columns:
    if df[column].dtype==np.number:
        continue
    df[column]=LabelEncoder().fit_transform(df[column])



#split the data into independent(X) dataset(the features) and dependent(y) data set(the target)

X= df.drop(['classification'],axis=1)
y=df['classification']

#Feature Scaling
#min-max scaler method which scales the dataset so that all input features lie between 0 to 1
x_scaler=MinMaxScaler()
x_scaler.fit(X)
column_names=X.columns
X[column_names]=x_scaler.transform(X)

#split the data into 80% taining and 20% testing & shuffle
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)

#build the model
model=Sequential()
model.add(Dense(256,input_dim=len(X.columns),kernel_initializer=k.initializers.random_normal(seed=13),activation='relu'))
model.add(Dense(1,activation='hard_sigmoid'))

#complie the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#train the model
history=model.fit(X_train,y_train,epochs=2000,batch_size=X_train.shape[0])

model.save('ckd.model')

#visualize the model loss and accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])

plt.title('model accuracy & loss')
plt.ylabel('accuracy and loss')
plt.xlabel('epoch')

pred=model.predict(X_test)
pred = [1 if y>=0.5 else 0 for y in pred]
print('Original : {0}'.format(", ".join(str(x) for x in y_test)))
print('Predicted : {0}'.format(", ".join(str(x) for x in pred)))


from sklearn import metrics
print("Accuracy Score: with ANN model")