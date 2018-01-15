
# coding: utf-8

# In[44]:

import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import datasets, linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

shuffle(diabetes['data'])
shuffle(diabetes['target'])

# We ll be using only 1 feature out of 10
X = diabetes.data[:, np.newaxis, 2]

type(X)


X_test = X[:20]
X_train = X[20:]
X_train


y_test = diabetes.target[:20]
y_train = diabetes.target[20:]


regr = linear_model.LinearRegression()


regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='red')
plt.plot(X_test, y_pred,  color='black')
plt.show()


# In[ ]:



