import glob
from keras.models import Sequential,load_model
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import matplotlib.pyplot as plt
import keras as k
df=pd.read_csv("chronicdisease.csv")
columns_to_retain=['sg','al','sc','hemo','pcv','wbcc','rbcc','htn','classification']
df=df.drop(['id',"'age'", "'bp'","'su'", "'rbc'", "'pc'", "'pcc'",
       "'ba'", "'bgr'", "'bu'","'sod'", "'pot'",
       "'dm'", "'cad'", "'appet'",
       "'pe'", "'ane'", 'Unnamed: 26'],axis=1)
df=df.dropna(axis=0)
#Transform the non-numeric data in the columns
for column in df.columns:
    if df[column].dtype==np.number:
        continue
    df[column]=LabelEncoder().fit_transform(df[column])
#split the data into independent (x) dataset(the features) and dependent (y) dataset(the target)
X=df.drop(["'class'"],axis=1)
y=df["'class'"]
#Feature Scaling
#min-max scaler method scales the data set so that all the input lie between 0 and 1
x_scaler=MinMaxScaler()
x_scaler.fit(X)
column_names=X.columns
X[column_names]=x_scaler.transform(X)
#split data into 80% training and 20% testing
X_train, X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
#Build the model
model=Sequential()
model.add(Dense(256,input_dim=len(X.columns), kernel_initializer=k.initializers.random_normal(seed=13), activation='relu'))
model.add(Dense(1,activation='hard_sigmoid'))
#Compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#train the model
history=model.fit(X_train,y_train,epochs=2000,batch_size=X_train.shape[0])
#save the model
model.save('ckd.model')
#visualize the model loss and accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy & loss')
plt.ylabel('accuracy & loss')
plt.xlabel('epoch')

