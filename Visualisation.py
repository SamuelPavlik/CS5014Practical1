#!/usr/bin/env python
# coding: utf-8

# # CS5014 Practical 1

# 150015752
# 
# 28.2.2019

# In[1]:


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

NUM_OF_REPS = 50


# ## Solution

# ### Loading and cleaning the data
# 

# We load the data using pandas library which can also store metadata such as column names. Additionally, pandas'
# DataFrame objects are easy to manipulate and slice.
# 

# In[2]:


dataset = pd.read_csv('ENB2012_data.csv')


# We slice the data to feature values and target values. In this case, feature values are all columns from X1 to X8 and
# target values are columns Y1 and Y2.

# In[3]:


X = dataset.loc[:, 'X1':'X8']
y = dataset.loc[:,'Y1':'Y2']


# ### Analysing and visualising the data

# To better visualize the relation between feature and target values, we implement a function that plots a specified
# target column as a function of each feature column separately.
# 

# In[81]:


def plotCorr(X, y, column_names, res, y_predicts=None):
    # Plot outputs
    num_of_features = X.shape[1]
    height = num_of_features*3
    plt.subplots(figsize=(15,height))
    if type(X) == np.ndarray:
        X_vals = X
        y_vals = y
    else:
        X_vals = X.values
        y_vals = y.values

    index = 1
    for col in range(0, num_of_features):
        plt.tight_layout()
        plt.subplot(num_of_features, 2, index).set_title("Real Y" + str(res + 1) + 
                                                         " values for " + column_names[col])
        plt.scatter(X_vals[:, col], y_vals[:, res])
        index+=1

        if y_predicts is not None:
            plt.subplot(num_of_features, 2, index).set_title("Predicted Y" + str(res + 1) + 
                                                             " values for " + column_names[col])
            plt.scatter(X_vals[:, col], y_predicts[:,res], color='black')
            index+=1
    plt.show()


# #### Relation plots of feature values for heating load

# In[82]:

print("\nFeature data plotted for Y1")
plotCorr(X, y, X.columns, 0)


# #### Relation plots of feature values for cooling load

# In[83]:

print("\nFeature data plotted for Y1")
plotCorr(X, y, X.columns, 1)


# Particularly for columns X6 and X8, that is Orientation and Glazing area distribution, we may notice that the
# distribution is very random. We might display the impact all the features have on HL and CL by a Pearson's
# correlation. Pearson's correlation is a number between -1 and 1 that indicates to what extent are 2 variables
# linearly related. 1 indicates total positive linear correlation, 0 no linear correlation and -1 total negative
# correlation. We implement a function that plots Pearson's correlation for each feature to both HL and CL.
# 

# In[66]:


def getPearsonPlot(X, y):
    num_of_features = X.shape[1]
    plt.subplots(figsize=(15,3))
    for target in range(0,2):
        vals = []
        for col in range(0, num_of_features):
            # xs.append( col)
            vals.append(np.abs(pearsonr(X.values[:,col], y.values[:,target])[0]))

        plt.subplot(1, 2, target + 1).set_title("Pearson's correlation coefficient for Y" + str(target + 1))
        barlist = plt.bar(X.columns, vals)
        vals_np = np.array(vals)
        min_val = vals_np.min()
        barlist[vals.index(min_val)].set_color('r')
    plt.show()


# In[69]:

print("\nPearson's correlation plots")
getPearsonPlot(X, y)


# As suspected, both X6 and X8 have a relatively low impact on the target values. These features may have impact on
# residential building in other areas, but with the dataset we're given they do not and we might create an additional
# dataset without these 2 columns, to test if there's any change in a regressor accuracy.

# In[11]:


def plotHists(X, column_names):
    # Plot outputs
    num_of_features = X.shape[1]
    height = num_of_features*3
    plt.subplots(figsize=(15,height))
    if type(X) == np.ndarray:
        X_vals = X
    else:
        X_vals = X.values

    index = 1
    for col in range(0, num_of_features):
        plt.subplot(num_of_features, 2, index).set_title(column_names[col])
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.tight_layout()
        # plt.hist(num_of_features, 2, index).set_title("Real Y" + str(res + 1) + 
        #                                          " values for " + column_names[col])
        plt.hist(X_vals[:, col])
        index+=1
    plt.show()



