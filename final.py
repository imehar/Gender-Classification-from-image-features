#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import numpy as np
import pickle
import face_recognition as fr
from sklearn.linear_model import LogisticRegression
import cv2


# In[2]:


model = pickle.load(open('model.sav','rb'))


# In[3]:


val = {
    0:'FEMALE',
    1:'MALE',
    2:'Undetermined',
}


# In[4]:


cap = cv2.VideoCapture(0)
cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
count = 0
while(True):
    if count ==1:
        count = 0
        continue
    ret, frame = cap.read()
    frame = cv2.resize(frame, (240, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("temp.jpg",gray)
    encode = fr.face_encodings(fr.load_image_file('temp.jpg'))
    if len(encode)>=1:
        encode = encode[0]
    else:
        continue
    np_encode = np.array(encode).reshape(1,-1)
    y = model.predict(np_encode)
    y = int(y)
    gender = val[y]
    cv2.putText(frame, gender, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
    cv2.imshow("output",frame)
    cv2.waitKey(1)
    if 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
    count+=1


# In[5]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




