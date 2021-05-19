#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


mushroom = {'number': {'e': 0, 'p':1}, 'name': {'e': 'Edible', 'p': 'Poisonous'}}

scent = {'number': {'a' : 0,'l' : 1,'c' : 2,'y' : 3,'f' : 4,'m' : 5,'n' : 6,'p' : 7,'s' : 8},
             'name': {'a' : 'Almond','l' : 'Anise','c' : 'Creosote','y' : 'Fishy',
                      'f' : 'Foul','m' : 'Musty','n' : 'None','p' : 'Pungent','s' : 'Spicy'}}

color = {'number': {'k':0,'n':1,'b':2,'h':3,'g':4,'r':5,
         'o':6,'p':7,'u':8,'e':9,'w':10,'y':11},
             'name': {'k':'Black','n':'Brown','b':'Buff','h':'Chocolate','g':'Gray','r':'Green',
         'o':'Orange','p':'Pink','u':'Purple','e':'Red','w':'White','y':'Yellow'}}

cols = ['Class Type', 'Odor', 'Gill Color']
colname = ['Class Name', 'Odor Name', 'Gill Color Name']

df = pd.read_table('agaricus-lepiota.data', delimiter=',', header=None, usecols=[0,5,9])
df.columns = cols

df["Class Type"] = df["Class Type"].map(mushroom['number'])
df["Odor"] = df["Odor"].map(scent['number'])
df["Gill Color"] = df["Gill Color"].map(color['number'])
df.head()


# In[23]:


dummycols = ['Odor', 'Gill Color']
X = pd.get_dummies(data=df, prefix=dummycols, prefix_sep=" ", columns=dummycols)
y = X['Class Type']
X.head()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[29]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
print({linreg.intercept_: linreg.coef_})


# In[30]:


y_pred = linreg.predict(X_test)
print(metrics.mean_squared_error(y_test, y_pred))


# In[40]:


from sklearn import svm
from sklearn.model_selection import cross_val_score

svt = svm.SVC(probability=True, random_state=0)
svt.fit(X,y)
cross_val_score(svt, X, y, scoring='neg_mean_squared_error')


# In[41]:


X = pd.get_dummies(data=df, columns=['Odor'])
y = X['Class Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

print((metrics.mean_squared_error(y_test, y_pred)))


# In[43]:


svt = svm.SVC(probability=True, random_state=0)
svt.fit(X,y)
cross_val_score(svt, X, y, scoring='neg_mean_squared_error')


# In[45]:


X = pd.get_dummies(data=df, columns=['Gill Color'])
y = X['Class Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print((metrics.mean_squared_error(y_test, y_pred)))


# In[46]:


svt = svm.SVC(probability=True, random_state=0)
svt.fit(X,y)
cross_val_score(svt, X, y, scoring='neg_mean_squared_error')


# In[ ]:


#Based on these comparisons, the Gill Color appears to be the better predictor on knowing if a mushroom is edible or poisonous.

