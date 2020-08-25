#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[77]:


dataset = pd.read_csv('sonar.csv',header=None)
dataset.head()

# In[84]:


X = dataset.iloc[:,0:60].values
y = dataset.iloc[:,60].values


# In[82]:


X


# In[80]:


y


# In[79]:


dataset.describe()


# In[49]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
y=encoder.fit_transform(y)
y


# In[6]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)


# In[36]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[50]:


import keras
from keras.models import Sequential
from keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(32, input_dim=60, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))


# In[51]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[58]:


history=model.fit(X_train, y_train, epochs=1000, batch_size=10)


# In[103]:


y_pred = model.predict(X_test)>0.5
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


# In[99]:


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a)


# In[100]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred,test)
print('confusion_matrix is:', cm)


# In[101]:


cm


# In[62]:


history=model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=10)


# In[75]:


#import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[76]:


plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

