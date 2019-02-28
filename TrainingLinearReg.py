import warnings
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

warnings.filterwarnings("ignore")

NUM_OF_REPS = 50

dataset = pd.read_csv('ENB2012_data.csv')
X = dataset.loc[:, 'X1':'X8']
y = dataset.loc[:,'Y1':'Y2']
X_reduced = dataset[['X1','X2','X3','X4','X5','X7']]

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


# The next step is to split the dataset to training and testing sets. We'll train on training dataset and evaluate the
# current model settings by k-fold cross validation. This technique is considered to be a very precise evaluation, but
# it is not used that often due to its computational expensiveness. The model is trained k times and tested on the
# hold-out set and can therefore take a while with more complex datasets. However, our dataset is actually not that
# large nor are the models that performance-heavy, and hence we can afford it. To get a more accurate result, we repeat
# this evaluation 50 times and get a mean over the results. We do not test on the testing set until we've found the
# optimal model with k-fold cross validation and want to evaluate the model on an unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(X_reduced, y, test_size=0.2)


# In[14]:


def rep_cross_val_score(model, X, y, cv, scoring, reps):
    sum = 0
    for i in range(0,reps):
        sum += cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
    return sum / reps


# ### Training and evaluating regression models
#

# #### Linear Regression
#

# With the data splitted, we can train a simple linear regression model without any additional preprocessing and
# evaluate it to see how well the model performs.
#

# In[71]:


lin_reg = LinearRegression()
error = rep_cross_val_score(lin_reg, X_train, y_train, 4, "neg_mean_squared_error", NUM_OF_REPS).mean()
print("\nLinear Regression with default parameters")
print("Mean squared error: " + str(-error))


# To judge this result, we need to know how large the error is relative to the target values. Therefore, we get min and
# max values for both target columns to see this. This might not be be the most precise way, but it only serves to
# demonstrate roughly rounded error of the model.

# In[16]:


def printError(y, error):
    min = y.min()
    max = y.max()

    print("MIN")
    print(min)
    print("\nMAX")
    print(max)
    print()
    diff0 = max[0] - min[0]
    diff1 = max[1] - min[1]
    print('Y1 (max - min) = ' + str(diff0))
    print('Y2 (max - min) = ' + str(diff1))
    sqrt_error = sqrt(-error)
    print("\nRoot mean squared error relative to the spread of Y1: " +
            str(round(sqrt_error, 2)) + " / " + str(diff0) + " = " + str(round(sqrt_error / diff0, 2)))
    print("\nRoot mean squared error relative to the spread of Y2: " +
            str(round(sqrt_error, 2)) + " / " + str(diff1) + " = " + str(round(sqrt_error / diff1, 2)))


# In[17]:

print("\nError statistics")
printError(y, error)


# The model would have a deviation of around 3.07 which is 8% from the spread of the both target columns. Now, we can
# try the reduced dataset, to see if removing the 2 feature columns helps in any way.
#

# In[75]:

print("\nLinear Regression with the reduced dataset")
lin_reg = LinearRegression()
error = rep_cross_val_score(lin_reg, X_red_train, y_red_train, 4, "neg_mean_squared_error", NUM_OF_REPS).mean()
print("Mean squared error with the reduced set: " + str(-error))


# The result demonstrates that removing the columns does not make a large difference in the linear regression model,
# however, we might keep the reduced dataset for other models, where it might.
#
# To improve performance and possibly an accuracy of a model we need to scale the features. The 2 main options for
# scaling are MinMaxScaler and StandardScaler. MinMaxScaler scales and shifts the values to range between 0 and 1,
# whereas StandardScaler centres the values to 0 by removing the mean and divides by the variance to achieve unit
# variance. StandardScaler should be applied to data which is normally distributed, and can handle outliers. However,
# if the data is not normally distributed, MinMaxScaler is better. The disadvantage is its sensitivity to outliers. To
# decide which scaler to use, we need to plot histograms to see the features' distributions.

# In[19]:


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


# In[20]:

print("\nHistograms for the Linear Regression model")
plotHists(X, X.columns)


# We can conclude from the histograms that the data is not distributed normally, and therefore MinMaxScaler should be
# better.
#

# In[84]:


lin_reg = LinearRegression(normalize=False)
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X_train)
error = rep_cross_val_score(lin_reg, X_minmax, y_train, cv=4, scoring="neg_mean_squared_error", reps=NUM_OF_REPS).mean()
print("\nLinear Regression with scaling")
print("Mean squared error after scaling: " + str(-error))


# Ridge is good default, Lasso and Elastic Net are useful when we presume that only few features are relevant. Since
# most of are features are relevant, that is 6 of them have Pearson's coefficient above 0.26, we use Ridge. An
# important hyperparameter in Ridge is $\alpha$. The higher it is, the more regularized model we're getting. To find
# right $\alpha$, we implement a method that prints out mean squared error for Ridge models with alpha from given
# minimum to given maximum.

# In[92]:


def frange(x, y, jump):
  while x < y:
    yield round(x,2)
    x += jump

def printAlphas(min, max, step):
    for i in frange(min, max, step):
        ridge = Ridge(alpha=i)
        ridge_score = rep_cross_val_score(ridge, X_minmax, y_train, cv=4,
                                          scoring="neg_mean_squared_error", reps=NUM_OF_REPS).mean()
        print("alpha " + str(i) + " => " + str(ridge_score))


# In[93]:

print("\nLinear Regression with regularisation")
print("Finding optimal alpha")
printAlphas(0.01, 0.1, 0.01)


# Suprisingly, regularization only makes the model worse. This means that we're already underfitting, which might be
# due to the lack of data or due to the linear regression not being quite right for this task. Now we have enough
# information to evaluate the model. The model performs best without any scaling or regularization, that is with just
# the basic linear regression model. We can therefore evaluate it on the testing set.
#

# In[24]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_predicts = lin_reg.predict(X_test)
print("\nFinal Linear Regression evaluation")
print("Mean squared error: " + str(mean_squared_error(y_test, y_predicts)))


# Additionally, we might demonstrate the accuracy of the linear model visually using our previously implemented function.

# In[95]:

print("\nFeature data plotted for Y1")
plotCorr(X_test, y_test, X.columns, 0, y_predicts)


# In[96]:

print("\nFeature data plotted for Y2")
plotCorr(X_test, y_test, X.columns, 1, y_predicts)
