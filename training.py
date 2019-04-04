#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow
import os
import dlib
import dlib.cuda as cuda
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import pickle
import os,os.path
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skimage import color
from skimage import io
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score


# In[ ]:


#!pip install face_recognition


# In[ ]:


# dlib.DLIB_USE_CUDA=1
# dlib.USE_AVX_INSTRUCTIONS=1
import face_recognition as fr


# In[ ]:


path = os.getcwd()


# In[ ]:


#os.listdir(path)


# In[ ]:


list_data = ['final_frontal0.csv',
 'final_frontal1.csv',
 'final_frontal2.csv',
 'final_frontal3.csv',
 'final_frontal4.csv']


# In[ ]:


#data = pd.read_csv(path+'final_frontal0.csv')
#data.head()


# In[ ]:


#data = data[['user_id' ,'original_image',  'gender' ,'age']]
#data.head()


# In[ ]:


def feature_extract(data):
    file_path = path + 'file/'
    count = 0
    for index,row in data.iterrows():
        if os.path.isfile(file_path + row['user_id'] +'/' + row['original_image']):
            encode = fr.face_encodings(fr.load_image_file(file_path + row['user_id'] +'/' + row['original_image']))
            if len(encode)>=1:
                encode = encode[0]
            else:
                continue
            if count == 0:
                temp = np.array([row['user_id'],row['original_image'],row['gender'],row['age']])
                temp = np.append(temp,encode)
                df = pd.DataFrame(temp).T
                count +=1
            else:
                temp = np.array([row['user_id'],row['original_image'],row['gender'],row['age']])
                temp = np.append(temp,encode)
                df = df.append(pd.DataFrame(temp).T ,ignore_index=True)           
    return df


# In[ ]:


#data.shape


# In[ ]:


#final_df = feature_extract(data)


# In[ ]:


#final_df.to_csv(path + 'feature0.csv')


# In[ ]:


for i in range(len(list_data)):
    #print(list_data[i])
    data = pd.read_csv(path+ list_data[i])
    data = data[['user_id' ,'original_image',  'gender' ,'age']]
    temp_df = feature_extract(data)
    #print(temp_df.shape)
    temp_df.to_csv(path + 'feature' + str(i) + '.csv')
    if i==0:
        final_df = temp_df.copy()
    else:
        final_df = final_df.append(temp_df,ignore_index=True)
    #print(final_df.shape)


# In[ ]:


#final_df.shape


# In[ ]:


del final_df


# In[ ]:


for i in range(len(list_data)):
    read_df = pd.read_csv(path + 'feature' + str(i)+'.csv')
    if i==0:
        final_df = pd.DataFrame(read_df)
    else:
        final_df = final_df.append(read_df,ignore_index=True)
    


# In[ ]:


#final_df.head()


# In[ ]:


trial_df = final_df.copy()
#trial_df.shape


# In[ ]:


img_df = trial_df[['0' , '1' , '2' , '3']]


# In[ ]:


trial_df = trial_df.drop(['0','1','2','3'],axis=1)


# In[ ]:


#trial_df.shape


# In[ ]:


#img_df.head()


# In[ ]:


y = img_df.copy()


# In[ ]:


y = y['2']


# In[ ]:


#y.unique()


# In[ ]:


#y.shape


# In[ ]:


from random import randint
y = y.fillna(randint(0,2))


# In[ ]:


y = y.replace('f',0)
y = y.replace('m',1)
y = y.replace('u',2)


# In[ ]:



#y.unique()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(trial_df, y,test_size=0.3, random_state=17)


# In[ ]:


X_train.shape , y_train.shape


# In[ ]:


logreg = LogisticRegression(random_state=17,multi_class='auto')
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))


# In[ ]:


clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = logreg.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[ ]:





# In[ ]:


penalty = ['l1', 'l2']
para_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 
hyperparameters = dict(C=para_C, penalty=penalty)
# para_C


# In[ ]:


logreg = LogisticRegression(random_state=17,multi_class='auto')
clf = GridSearchCV(logreg, hyperparameters, cv=5, verbose=0)


# In[ ]:


best_model = clf.fit(X_train , y_train)


# In[ ]:


best_model.best_score_


# In[ ]:


best_model.best_estimator_.fit(X_train,y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(best_model.best_estimator_.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(best_model.best_estimator_.score(X_test, y_test)))


# In[ ]:


model_file = path + 'model.sav'


# In[ ]:


pickle.dump(best_model.best_estimator_,open(model_file,'wb'))


# In[ ]:


model = pickle.load(open(model_file,'rb'))


# In[ ]:




