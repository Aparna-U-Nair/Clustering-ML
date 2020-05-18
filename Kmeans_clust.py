# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:01:40 2020

@author: Aparna
"""

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
X = data.iloc[:,3:].values

#choose the optimal 'k' using elbow method
#‘k-means++’ : selects initial cluster centers for k-means clustering in a smart way to 
#speed up convergence, avoids the random initialisation trap.
from sklearn.cluster import KMeans
#WCSS(Within Cluster Sum of Sqaures - metric for clustering algo)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init ='k-means++',max_iter = 300, n_init = 10,\
                    random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('ELBOW METHOD')
plt.xlabel('K values')
plt.ylabel('WCSS')
plt.show()

# k=5 optimum
kmeans = KMeans(n_clusters = 5,init ='k-means++',max_iter = 300, n_init = 10,\
                random_state= 0)
y = kmeans.fit_predict(X)

# plot the clusters
# X[:,0] or X[:,1] col0 in Xaxis and col1 in Yaxis
plt.scatter(X[y==0,0],X[y==0,1],s =100,c ='red', label = 'cluster 1')
plt.scatter(X[y==1,0],X[y==1,1],s =100,c ='blue', label = 'cluster 2')
plt.scatter(X[y==2,0],X[y==2,1],s =100,c ='green', label = 'cluster 3')
plt.scatter(X[y==3,0],X[y==3,1],s =100,c ='cyan', label = 'cluster 4')
plt.scatter(X[y==4,0],X[y==4,1],s =100,c ='magenta', label = 'cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,\
            c='yellow',label='Centroids')
plt.title('Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()