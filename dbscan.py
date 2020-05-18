# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:20:42 2020

@author: Aparna
"""

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:,-2:].values

#Algo to find optimum eps
#This will find the distance of each point to n nearest points, sort it and then plot.
#min_pts >= D+1 where D is the dimension.
nearest = NearestNeighbors(n_neighbors=2)
near_obj = nearest.fit(X)
dist, index = near_obj.kneighbors(X)
dist = np.sort(dist,axis=0)
plt.plot(dist)
plt.ylabel('eps')
plt.title('Optimum EPS Curve')

dbscan = DBSCAN(eps=3,min_samples=4)

model = dbscan.fit(X)

#Gives the list of clusters and respective points.
labels = model.labels_
#-1 indicates outliers, as they don't fall into any clusters. They are noise points.

#set ll have unique values only.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#This ll give you the indices of core points from the total lables, which also has noise pts.
core_pts = dbscan.core_sample_indices_

sample_core = np.zeros_like(labels,dtype='bool')
sample_core[core_pts] = True
print(len(sample_core))
print(len(sample_core[core_pts] == True))
print(silhouette_score(X,labels))

#Lets visualize
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive',
          'goldenrod', 'lightcyan', 'navy']
for i in set(labels):
    print(i, colors[i % len(colors)])
    
vect = np.vectorize(lambda x: colors[x % len(colors)])

plt.scatter(X[:,0], X[:,1], c = vect(labels))
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('DBSCAN CLUSTERING')
#Dark blue points are the noise (-1 label)



