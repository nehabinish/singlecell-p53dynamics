#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:02:09 2022

@author: nehabinish
"""
#%% IMPORT REQUIRED LIBRARIES

# Native libraries
import os
import math
import numpy as np
import logging

# math libraries
import pandas as pd
from scipy.integrate import simps
from scipy.signal import correlate


# plotting libraries
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
X = random_walks(n_ts=50, sz=32, d=1)

seed = 0
np.random.seed(seed)

#%% Input data sets - p53 dynamics

# --- Load input data into arrays ---
p53_0G  = pd.read_csv('p53_0G.csv'  , header=None, delimiter=",")
p53_2G  = pd.read_csv('p53_2G.csv'  , header=None, delimiter=",")
p53_4G  = pd.read_csv('p53_4G.csv'  , header=None, delimiter=",")
p53_10G = pd.read_csv('p53_10G.csv', header=None, delimiter=",")


#%% 

Tpoints_0G = p53_0G.shape[0]
Ncells_0G  = p53_0G.shape[1]

p53_0G = np.array(p53_0G)
#p53_0G = np.reshape(p53_0G, (Tpoints_0G, Ncells_0G, 1))

km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=10000, 
                      random_state=0)

y_pred = km.fit_predict(p53_0G)

plt.figure()
for yi in range(3):
    plt.subplot(3, 3, yi + 1)
    for xx in p53_0G[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, Ncells_0G)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")
        
# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=3,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(p53_0G)

for yi in range(3):
    plt.subplot(3, 3, 4 + yi)
    for xx in p53_0G[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, Ncells_0G)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
              transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DBA $k$-means")

# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=3,
                            metric="softdtw",
                            metric_params={"gamma": .01},
                            verbose=True,
                            random_state=seed)
y_pred = sdtw_km.fit_predict(p53_0G)

for yi in range(3):
    plt.subplot(3, 3, 7 + yi)
    for xx in p53_0G[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, Ncells_0G)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
              transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

        
plt.tight_layout()
plt.show()
        
