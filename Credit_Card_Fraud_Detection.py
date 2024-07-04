#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[125]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# In[126]:


#loading the dataset to a Pandas DataFrame
credit_card_data=pd.read_csv("D://Credit Card Detection System//creditcard.csv")


# In[127]:


#First five rows of dataset
credit_card_data.head()


# In[128]:


credit_card_data.tail()


# In[129]:


# Dataset information
credit_card_data.info()


# In[130]:


#checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[131]:


#Distribution of legit transaction & fraudulent transactions
credit_card_data['Class'].value_counts()


# This Dataset is highly unbalanced

# 0-->Normal Transaction
# 
# 1-->Fraudulent Transaction

# In[132]:


#Separating the data for analysis
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]


# In[133]:


print(legit.shape)
print(fraud.shape)


# In[134]:


#Statistical measures of the data- This will take amount column
legit.Amount.describe() 


# In[135]:


fraud.Amount.describe()


# In[136]:


#compare the values for both transactions
credit_card_data.groupby('Class').mean()


# Build a sample dataset containing similar distribution of normal transaction and Fraudalent transaction

# Number of Fraudulent Transaction-->492

# In[137]:


legit_sample=legit.sample(n=492)


# Concatenating two DataFrames

# In[138]:


new_dataset=pd.concat([legit_sample,fraud],axis=0)


# In[139]:


new_dataset.head()


# In[140]:


new_dataset.tail()


# In[141]:


new_dataset['Class'].value_counts()


# In[142]:


new_dataset.groupby('Class').mean()


# Splitting the data into Features & Targets

# In[143]:


X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']


# In[144]:


print(X)


# In[145]:


print(Y)


# To split the data into Training Data & Testing Data

# In[146]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[147]:


print(X.shape,X_train.shape,X_test.shape)


# Model Training

# Logistic Regression

# In[148]:


model = LogisticRegression()


# In[149]:


#training the Logistic Regression Model with Training Data
model.fit(X_train,Y_train)


# Model Evaluation

# Accuracy Score

# In[150]:


print('Accuracy on Training data: ', training_data_accuracy)


# In[151]:


#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[152]:


#Accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[153]:


print('Accuracy score on Test Data : ',test_data_accuracy)


# In[154]:


# Correlation matrix
corrmat = credit_card_data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




