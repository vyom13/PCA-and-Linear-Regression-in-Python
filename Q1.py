
# coding: utf-8

# In[18]:

import numpy as np
from numpy import genfromtxt
from numpy import linalg as lg
from numpy import linalg as LA
from matplotlib import pyplot as plt
import pandas as pd

def d_PCA(x):
    columnMean = x.mean(axis=0)
    columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    xMeanCentered = x - columnMeanAll
#use mean_centered data or standardized mean_centered data
    
    dataForPca = xMeanCentered

#get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca, rowvar=False)
#eigendecomposition of the covariance matrix
    eigenValues, eigenVectors = LA.eig(covarianceMatrix)
    II = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[II]
    eigenVectors = eigenVectors[:, II]
#get scores
    pcaScores = np.matmul(dataForPca, eigenVectors)
#collect PCA results
    pcaResults = {'data': x,
                         'mean_centered_data': xMeanCentered,
                         'PC_variance': eigenValues,'loadings': eigenVectors,
                         'scores': pcaScores}
    return pcaResults


data_1 = genfromtxt("C:/Users/vyoms/Desktop/linear_regression_test_data.csv", delimiter=",")

data = data_1[1:,1:3]
data_x = data_1[1:,1]
data_y = data_1[1:,2]
data_yth = data_1[1:,3]

myPCAResults = d_PCA(data)

X = data_x
Y = data_y

# Total no. of X values
m = len(X)

# Mean of X and Y values
mean_x = np.mean(X)
mean_y = np.mean(Y)

# calculating b1 and b2
numerator = 0
denominator = 0
for i in range(m):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
b1 = numerator / denominator
b0 = mean_y - (b1 * mean_x)

max_x = np.max(X) 
min_x = np.min(X)

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('scores plot')
ax.scatter(data_x, data_y, color='tomato')
ax.scatter(data_x,data_yth, color='dodgerblue')
ax.plot([0,200*myPCAResults['loadings'][0,0]], [0, 200*myPCAResults['loadings'][1,0]],
            color='black', linewidth=3)
plt.plot(x, y, color='plum', linewidth=3, label='Regression Line')
plt.xlim(-4, 4), plt.ylim(-4, 4)


plt.show()


# In[9]:

b0


# In[10]:

b1


# 

# In[ ]:



