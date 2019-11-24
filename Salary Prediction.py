#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as py
import pandas as pd


# In[5]:


dataset = pd.read_csv(r'C:\Users\Admin\Documents\Salary_Data.csv')


# In[6]:


dataset.head()


# In[65]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# In[66]:


X


# In[79]:


y


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


X_train,x_test,Y_train,y_test = train_test_split(X,y,test_size =1/3,random_state =0)    


# In[75]:


from sklearn.linear_model import LinearRegression


# In[76]:


model = LinearRegression()


# In[77]:


model.fit(X_train,Y_train)


# In[78]:


y_pred= model.predict(x_test)


# In[80]:


#visulalizing the training dataset


# In[82]:


#Visualizing the training data set results 
py.scatter(X_train,Y_train,color='red')


# In[83]:


py.plot(x_test,y_test,color='Blue')


# In[84]:


py.plot(x_test,y_pred,color='red')


# In[85]:


y_pred


# In[90]:


py.title('salary VS Experience')
py.xlabel('Years of experience')
py.ylabel('Salary')
py.scatter(x_test,y_pred,color='red')
py.scatter(x_test,y_test,color='blue')


# In[ ]:




