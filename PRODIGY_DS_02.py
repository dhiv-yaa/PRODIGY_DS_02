#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('Train.csv')


# In[5]:


print(df.head())


# In[6]:


print("Missing values in each column:")
print(df.isnull().sum())


# In[7]:


df = df.drop_duplicates()
print("\nNumber of rows after removing duplicates:", df.shape[0])


# In[11]:


print("\nSummary Statistics:")
print(df.describe())


# In[16]:


print("\nHistograms:")
df.hist(bins=20, figsize=(10, 10))
plt.suptitle('Histograms of EDA analysis')
plt.show()


# In[17]:


print("\nCorrelation Matrix:")
corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, cmap='coolwarm')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:




