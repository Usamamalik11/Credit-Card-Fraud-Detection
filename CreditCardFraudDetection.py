#!/usr/bin/env python
# coding: utf-8

# In[17]:


import sys
import numpy
import pandas
import matplotlib
import scipy
import sklearn
import seaborn


#printing the versions
print('sys:{}'.format(sys.version))
print('numpy:{}'.format(numpy.__version__))
print('pandas:{}'.format(pandas.__version__))
print('matplotlib:{}'.format(matplotlib.__version__))
print('scipy:{}'.format(scipy.__version__))
print('sklearn:{}'.format(sklearn.__version__))
print('seaborn:{}'.format(seaborn.__version__))


# In[18]:


# Specifying the aliases
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


#Load the dataset from the csv file
data  = pd.read_csv('creditcard.csv')


# In[20]:


#exploring the dataset
print(data.columns)


# In[21]:


#Shape of the dataset
print (data.shape)


# In[22]:


#Shows the characteristics of each parameter/column
print(data.describe())


# In[23]:


#We will work only on a fraction of dataset for computational reasons
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)


# In[24]:


#plot histogram of each parameter
data.hist(figsize=(20,20))
plt.show()


# In[25]:


#Determine the fraud and valid cases in the dataset
Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_frac = len(Fraud)/float(len(Valid))
print(outlier_frac)
print('Fraud Cases:{}'.format(len(Fraud)))
print('Valid Cases:{}'.format(len(Valid)))


# In[26]:


#Correlation Matrix
corrmat = data.corr()
Fig = plt.figure(figsize= (12,9))

sns.heatmap(corrmat,vmax = .8,square = True)
plt.show()


# In[27]:


#Getting all the columns in the list
columns = data.columns.tolist()

#Filtering the columns we don't want
columns = [c for c in columns if c not in ['Class']]

#The variable which we'll be predicting on
target = "Class"


X = data[columns]
Y = data[target]

#Print the shapes of X and Y
print(X.shape)
print(Y.shape)


# In[28]:


#Import the algorithms
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#Initializing random state
state = 1

#Putting the classifiers inside a dictionary
classifiers = {"IsloationForest": IsolationForest(max_samples = len(X),contamination = outlier_frac, random_state = state),
              "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20,contamination = outlier_frac)}


# In[30]:


#Fit the model
n_outliers  =len(Fraud)

#Running the loop
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    #Reshaping the prediction values for 0 if valid and 1 for fraudelent
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    
    n_errors =(y_pred != Y).sum()
    
    #Run classification metrics
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
    
        
        


# In[ ]:




