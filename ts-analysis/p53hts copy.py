#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 01:11:50 2022

@author: nehabinish
"""

"""
Plot p53 and p21 Intensity Levels

"""

# Native libraries
import os
import math
import logging

#  Essential libraries
import pandas as pd
from scipy.integrate import simps
from scipy.signal import correlate
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from tslearn.generators import random_walks

# Algorithms"
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans

import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess

from scipy.signal import find_peaks
import pandas as pd

#%% NORMALISATION FUNCTIONS

def normalization(x):
    
    Tpoints = x.shape[0]
    Ncells  = x.shape[1]
    
    z = np.zeros((Tpoints,Ncells))
    
    for i in range(Ncells):
        y  = x[:,i]
        minim = min(y)
        maxim = max(y)
        
        for j in range(Tpoints):            
            z[j,i]=(y[j]-minim)/(maxim-minim)
        
    return z

#%% LOAD DATASET

# Load time series datasets for different doses - p53
p53_0G  = pd.read_csv('p53_0G.csv'  , header=None, delimiter=",")
p53_2G  = pd.read_csv('p53_2G.csv'  , header=None, delimiter=",")
p53_4G  = pd.read_csv('p53_4G.csv'  , header=None, delimiter=",")
p53_10G = pd.read_csv('p53_10G.csv' , header=None, delimiter=",")

# Load time series datasets for different doses - p21
p21_0G  = pd.read_csv('p21_0G.csv'  , header=None, delimiter=";")
p21_2G  = pd.read_csv('p21_2G.csv'  , header=None, delimiter=";")
p21_4G  = pd.read_csv('p21_4G.csv'  , header=None, delimiter=";")
p21_10G = pd.read_csv('p21_10G.csv' , header=None, delimiter=";")



# normalisation
p21_0G  = normalization(np.array(p21_0G.T))
ID0_p21 = np.arange(0,len(p21_0G),1)
p21_2G  = normalization(np.array(p21_2G.T))
ID2_p21 = np.arange(0,len(p21_2G),1)
p21_4G  = normalization(np.array(p21_4G.T))
ID4_p21 = np.arange(0,len(p21_4G),1)
p21_10G  = normalization(np.array(p21_10G.T))
ID10_p21 = np.arange(0,len(p21_10G),1)
p53_0G = normalization(np.array(p53_0G.T))
ID0 = np.arange(0,len(p53_0G),1)
p53_2G = normalization(np.array(p53_2G.T))
ID2 = np.arange(0,len(p53_2G),1)
p53_4G = normalization(np.array(p53_4G.T))
ID4 = np.arange(0,len(p53_4G),1)
p53_10G = normalization(np.array(p53_10G.T))
ID10 = np.arange(0,len(p53_10G),1)


# label lists
p53    = [p53_0G,p53_2G,p53_4G,p53_10G]
IDp53  = [ID0, ID2, ID4, ID10]
p21    = [p21_0G,p21_2G,p21_4G,p21_10G]
IDp21  = [ID0_p21, ID2_p21, ID4_p21, ID10_p21]

#%%
# max-height of peaks
ht  = [0.5,0.3,0.3,0.5]
for i in  range(4):
  
    fig = plt.figure(figsize=(4,8), edgecolor='black')  
    plt.ylabel('Cell #')
    dosep21  = p21[i]
    dosep53  = p53[i]
    labelp21 = IDp21[i] 
    labelp53 = IDp53[i]
    tpeak = []
    
    for j in range(len(labelp53)):
        
        # find high intensity peaks for sorting
        peaks,_ = find_peaks(dosep53[j][-48:], height=ht[i], prominence=0.0)
        
        #print(len(peaks))   
        if len(peaks) == 0:
            peaks = 0
        tpeak.append(peaks)

    for k in range(len(tpeak)):
        if(np.any(tpeak[k] != 0)):
            tpeak[k] = 10
        
    tpeak    = tuple(tpeak)
    dosep21  = tuple([tuple(e) for e in dosep21])
    dosep53  = tuple([tuple(e) for e in dosep53])
    
    # sort time-series
    tpeak,dosep21,labelp21 = zip(*sorted(zip(tpeak,dosep21,labelp21)))          
    print(tpeak)     

    # axesp53.append(fig.add_subplot(1, 4, i+1) )
    # im = plt.imshow(dosep53, vmin=minminp53, vmax=maxmaxp53, interpolation='nearest', aspect='auto')
    # im.set_cmap("inferno")
    # plt.xlabel('Time(h)')
    # positions=(0,100,200)
    # label=(0,50,100)
    # plt.xticks(positions,label)
    
    
    # axesp21.append(fig.add_subplot(1, 4, i+1) )
    # im = plt.imshow(dosep21, vmin=minminp21, vmax=maxmaxp21, interpolation='nearest', aspect='auto')
    # im.set_cmap("inferno")
    # plt.xlabel('Time(h)')
    # positions=(0,100,200)
    # label=(0,50,100)
    # plt.xticks(positions,label)
    
    im = plt.imshow(dosep21, interpolation='nearest', aspect='auto')
    im.set_cmap("hot")
    plt.xlabel('Time(h)')
    positions=(0,100,200)
    label=(0,50,100)
    plt.xticks(positions,label)
    
    ax=plt.gca()
    divider=make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb=plt.colorbar(cax=cax)
    
    plt.savefig('p21{}.svg'.format(str(i)))

