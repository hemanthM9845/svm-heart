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


# In[33]:


df.AHD.replace(("Yes","No"),(1,0),inplace=True)


# In[34]:


df.head()


# In[35]:


df['Ca'].fillna('0.0',inplace=True)


# In[36]:


df.head()


# In[37]:


df.corr()


# In[38]:


x=df[["Age","Sex","Chol"]]
y=df[["AHD"]]


# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)


# In[40]:


x_train


# In[41]:


x_test


# In[42]:


y_train


# In[43]:


y_test


# In[44]:


from sklearn.svm import SVC
clf=SVC(kernel='linear')
clf.fit(x_train,y_train)


# In[45]:


predictedOutput=clf.predict(x_test)


# In[46]:


predictedOutput


# In[47]:


from  sklearn.metrics import classification_report
print(classification_report(y_test,predictedOutput))


# In[ ]:




