# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:39:07 2020

@author: Aparna
"""

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:,3:].values


import scipy.cluster.hierarchy as sch
#method='ward' to minimise the within cluster variance (metric)
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Points')
plt.ylabel('Euclidean Distance')
plt.show()

#We found that the optimal cluster size  = 5 because largerst distance for vertical line
#not crossing any horizontal line comes below or at 200 level.

from sklearn.cluster import AgglomerativeClustering
#affinity is the distance type used for linkage.
amc = AgglomerativeClustering(n_clusters =5, affinity = 'euclidean',linkage = 'ward',)
y = amc.fit_predict(X)

#plot the clusters
plt.scatter(X[y==0,0],X[y==0,1],s =300,c='red',label = 'cluster0')
plt.scatter(X[y==1,0],X[y==1,1],s =300,c='green',label = 'cluster1')
plt.scatter(X[y==2,0],X[y==2,1],s =300,c='blue',label = 'cluster2')
plt.scatter(X[y==3,0],X[y==3,1],s =300,c='cyan',label = 'cluster3')
plt.scatter(X[y==4,0],X[y==4,1],s =300,c='magenta',label = 'cluster4')
plt.title('HC')
plt.legend()
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()