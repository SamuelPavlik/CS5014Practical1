import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")

NUM_OF_REPS = 50
dataset = pd.read_csv('ENB2012_data.csv')
X = dataset.loc[:, 'X1':'X8']
y = dataset.loc[:,'Y1':'Y2']
X_reduced = dataset[['X1','X2','X3','X4','X5','X7']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(X_reduced, y, test_size=0.2)

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

def frange(x, y, jump):
  while x < y:
    yield round(x,2)
    x += jump


# #### Gradient Boosting Regression


# We can look at the mean of 4-fold cross validation for both models with default values. It should be mentioned, that
# scikit's implementation of Gradient Boosting does not support multi-dimensional output, therefore we have to train on
# each column separately.
#

# In[27]:


def get_score(X_t, y_t):
    dec_tree = ensemble.GradientBoostingRegressor()
    rand_tree = ensemble.RandomForestRegressor()
    print("Gradient Boosting Tree Regressor for Y1: " + str(cross_val_score(dec_tree,
                                                                            X_t,
                                                                            y_t.values[:,0],
                                                                            cv=4,
                                                                            scoring='neg_mean_squared_error').mean()))
    print("Random Forest Regressor for Y1: " + str(cross_val_score(rand_tree,
                                                                    X_t,
                                                                    y_t.values[:,0],
                                                                    cv=4,
                                                                    scoring='neg_mean_squared_error').mean()))
    print("Gradient Boosting Tree Regressor for Y2: " + str(cross_val_score(dec_tree,
                                                                            X_t,
                                                                            y_t.values[:,1],
                                                                            cv=4,
                                                                            scoring='neg_mean_squared_error').mean()))
    print("Random Forest Regressor for Y2: " + str(cross_val_score(rand_tree,
                                                            X_t,
                                                            y_t.values[:,1],
                                                            cv=4,
                                                            scoring='neg_mean_squared_error').mean()))


# In[28]:

print("\nGradient Boosting Regression with default hyperparameters")
get_score(X_train, y_train)


# And the performance for the reduced dataset:

# In[97]:

print("\nGradient Boosting Regression for the reduced datase")
get_score(X_red_train, y_red_train)


# Gradient Boosting outperforms Random Forests with default values. In light of what we know, this is probably due to
# Random Forests underfitting, that is having a high bias. Gradient Booosting actually performs worse with the reduced
# dataset. This might indicate that the removed features still have a slight impact on the prediction and do not behave
# randomly. We can further decrease the error rate by tweeking hyperparameters. Similar to Linear Regression, Gradient
# Boosting has learning rate as a hyperparameter. We might want to find the optimal value for it by iteratively
# evaluating the model with different learning rates.

# In[30]:


def getLRPlot(X, y, target):
    xs = []
    vals = []
    for lr in frange(0.1,1.5, 0.1):
        xs.append(lr)
        vals.append(-cross_val_score(ensemble.GradientBoostingRegressor(learning_rate=lr),
                                        X,
                                        y.values[:,target],
                                        cv=4,
                                        scoring='neg_mean_squared_error').mean())

    plt.subplot().set_title('Mean squared error over different learning rates')
    barlist = plt.bar(xs, vals, width=0.07)
    vals_np = np.array(vals)
    min_val = vals_np.min()
    barlist[vals.index(min_val)].set_color('r')
    plt.show()

    return min_val


# In[31]:

print("\nLearning rate plot for Y1")
lr1 = getLRPlot(X_train, y_train, 0)


# In[32]:

print("\nLearning rate plot for Y2")
lr2 = getLRPlot(X_train, y_train, 1)


# We've found optimal learning rates for both values and these are used for the final evaluation on the testing set.
#

# In[33]:


lin_reg1 = ensemble.GradientBoostingRegressor(learning_rate=lr1)
lin_reg2 = ensemble.GradientBoostingRegressor(learning_rate=lr2)
lin_reg1.fit(X_train, y_train.values[:,0])
lin_reg2.fit(X_train, y_train.values[:,1])
y_predicts1 = lin_reg1.predict(X_test)
y_predicts2 = lin_reg2.predict(X_test)
mse1 = mean_squared_error(y_test.values[:,0], y_predicts1)
mse2 = mean_squared_error(y_test.values[:,1], y_predicts2)

y_pred = np.concatenate((np.expand_dims(y_predicts1, axis=1), np.expand_dims(y_predicts2, axis=1)), axis=1)

print("\nFinal Gradient Boosting Regression Evaluation")
print("Mean squared error for Y1: " + str(round(mse1, 2)))
print("Mean squared error for Y2: " + str(round(mse2, 2)))


# Now we can demonstrate the improvement in predictions graphically, using the plotting function we implemented earlier.

# In[34]:

print("\nFeature data plotted for Y1")
plotCorr(X_test, y_test, X_test.columns, 0, y_pred)


# In[35]:

print("\nFeature data plotted for Y2")
plotCorr(X_test, y_test, X_test.columns, 1, y_pred)
