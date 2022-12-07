#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


df=pd.read_csv("C:/Users/user/Desktop/Heart.csv")


# In[12]:


df


# In[13]:


df.head()


# In[14]:


df.tail()


# In[15]:


df.describe


# In[16]:


df=df.rename(columns={"ChestPain":"CP"})


# In[17]:


df.head()


# In[18]:


df.AHD.replace(("yes","no"),(1,0),inplace=True)


# In[19]:


df.head()


# In[20]:


df['Ca'].fillna('0.0',inplace=True)


# In[21]:


df.head()


# In[22]:


df.corr()


# In[23]:


x=df[["Age","Sex","Chol"]]
y=df[["AHD"]]


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)


# In[25]:


x_train


# In[26]:


x_test


# In[27]:


y_train


# In[28]:


y_test


# In[29]:


from sklearn.svm import SVC
clf=SVC(kernel='linear')
clf.fit(x_train,y_train)


# In[30]:


predictedOutput=clf.predict(x_test)


# In[31]:


predictedOutput


# In[32]:


from  sklearn.metrics import classification_report
print(classification_report(y_test,predictedOutput))


# In[ ]:




