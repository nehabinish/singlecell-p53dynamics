#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 07:42:12 2022

@author: nehabinish
"""

"""
K-means correlation of signals
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

# Algorithms
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


#%% FUNCTIONS

# auto-correlation function
def acf(x, n_lags):
    return sm.tsa.stattools.acf(x, nlags=n_lags)
 
# plot time series 
def plot_df(df, lower, upper, process):
    n_rows = upper - lower
    fig, ax = plt.subplots(
        nrows=n_rows, ncols=1,
        sharex=True, sharey=False,
        figsize=(12, 2 * n_rows),
        tight_layout=True)
    fig.suptitle(f"TS {lower}-{upper-1}\n{process}")
    for i in range(n_rows):
        ts_idx = lower + i
        ax[i].plot(df.iloc[:, ts_idx])
        #ax[i].set_title(f"TS {ts_idx}")
    plt.show()
    
def normalization(x):
    
    x = np.array(x)
    
    Tpoints = x.shape[0]
    Ncells  = x.shape[1]   
    z = np.zeros((Tpoints,Ncells))  
    print(np.shape(x))
    
    for i in range(Ncells):
        y  = x[:,i]
        minim = min(y)
        maxim = max(y)
        
        for j in range(Tpoints):            
            z[j,i]=(y[j]-minim)/(maxim-minim)
        
    return z    

# plot two-dimensional heatmaps
def htsplot(hts, k, cond, name):
    
    # HEAT MAP PLOT
    
    rows = k
    cols = 1
    axes =[]
    min_ = []
    max_ = []

    for a in range(rows*cols):    
        b = np.array(hts[a]) 
        min_.append(np.min(b))
        max_.append(np.max(b))

    minmin = np.min(min_)
    maxmax = np.max(max_)    
    
    if(name=='p53'):  
        
        fig = plt.figure(figsize=(12,10), edgecolor='black')
        for a in range(rows*cols):
            print(a)
            
            # heat map
            b = np.array(hts[a]) 
            axes.append(fig.add_subplot(rows, cols, a+1))
            im = plt.imshow(b, vmin=minmin, vmax=maxmax, interpolation='nearest', aspect='auto')
            plt.tick_params(axis='both',which='both',top=False,left=False,right=False,labelleft=False)
            plt.ylabel('Cluster {}\n n = {} '.format(a+1, len(b)), rotation='horizontal', labelpad=45)
            plt.xticks([])
       
        # xticks
        positions=(0,50,100,150,200)
        label=(0,25,50,75,100)
        plt.xticks(positions,label)
        plt.xlabel('Time(h)')
        
        fig.tight_layout() 
        plt.subplots_adjust(hspace=0)     
        
        # colourbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='p53 normalised levels')
             
        plt.savefig('clustercount{}.pdf'.format(str(cond), str(k)))
        plt.savefig('clustercount{}.svg'.format(str(cond), str(k)))
        
            
        # dynamics 
        fig = plt.figure(figsize=(12,10), edgecolor='black')   
        for a in range(rows*cols):
            
            b = np.array(hts[a]) 
            for i in range(len(b)):
                ts = b[i]
                plt.subplot(rows, cols, a+1)
                plt.plot(ts, c="gray", alpha=0.4)
                plt.tick_params(axis='both',which='both',top=False,left=False,right=False,labelleft=False)
                plt.ylabel('Cluster {}\n n = {} '.format(a+1, len(b)), rotation='horizontal', labelpad=45)
                plt.xticks([])
                
            plt.plot(np.average(np.vstack(b),axis=0),c="red")
            
        # xticks
        positions=(0,50,100,150,200)
        label=(0,25,50,75,100)
        plt.xticks(positions,label)
        plt.xlabel('Time(h)')
        
        fig.tight_layout() 
        plt.subplots_adjust(hspace=0)     
             
        plt.savefig('dynamics{}.pdf'.format(str(cond), str(k)))
        plt.savefig('dynamics{}.svg'.format(str(cond), str(k)))
        
    if(name=='p21'):
        
        fig = plt.figure(figsize=(12,10), edgecolor='black')
        for a in range(rows*cols):
            b = np.array(hts[a]) 
            axes.append(fig.add_subplot(rows, cols, a+1) )
            im = plt.imshow(b, vmin=minmin, vmax=maxmax, interpolation='nearest', aspect='auto', cmap='gist_heat')
            plt.tick_params(axis='both',which='both',top=False,left=False,right=False,labelleft=False)
            plt.ylabel('Cluster {}\n n = {} '.format(a+1, len(b)), rotation='horizontal', labelpad=45)
            plt.xticks([])
         
        # xticks
        positions=(0,50,100,150,200)
        label=(0,25,50,75,100)
        plt.xticks(positions,label)
        plt.xlabel('Time(h)')
        
        fig.tight_layout() 
        plt.subplots_adjust(hspace=0)     
        
        # colourbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='p21 normalised levels')
  
        plt.savefig('p21-afmhot-cluster{}.pdf'.format(str(cond), str(k)))
        plt.savefig('p21-afmhot-cluster{}.svg'.format(str(cond), str(k)))
        
    plt.show()    
    
def acfplot(acf_loc, acf_label, k, cond):    
    
    # PLOTS
    
    rows = k
    cols = 1
    axes =[]

    fig = plt.figure(figsize=(12,10), edgecolor='black')
    for a in range(rows*cols):
        b = acf_label[a]
        axes.append(fig.add_subplot(rows, cols, a+1) )
        for j in range(len(b)):
            plt.plot(acf_loc[:, b])
            plt.tick_params(axis='both',which='both',top=False,left=False,right=False,labelleft=False)
            plt.ylabel('Cluster {}\n n = {}'.format(a+1, len(b)), rotation='horizontal', labelpad=45)
            plt.xticks([])
     
    # # xticks
    # positions=(0,50,100,150,200)
    # label=(0,25,50,75,100)
    # plt.xticks(positions,label)
    # plt.xlabel('Time(h)')
    
    fig.tight_layout() 
    plt.subplots_adjust(hspace=0)     
              
    plt.savefig('/Users/nehabinish/Nextcloud/scripts/clustering/corr-plot/normalised/{}/acf_cluster_k{}.pdf'.format(cond, str(k)))
    plt.savefig('/Users/nehabinish/Nextcloud/scripts/clustering/corr-plot/normalised/{}/acf_cluster_k{}.svg'.format(cond, str(k)))
    plt.show()    

 


#%% DATASET

# Load time series datasets for different doses 
p53_0G  = pd.read_csv('p53_0G.csv'  , header=None, delimiter=",")
p53_2G  = pd.read_csv('p53_2G.csv'  , header=None, delimiter=",")
p53_4G  = pd.read_csv('p53_4G.csv'  , header=None, delimiter=",")
p53_10G = pd.read_csv('p53_10G.csv' , header=None, delimiter=",")

Ncells = p53_0G.shape[1]
for i in range(Ncells):
    ind = p53_0G[i]
    p53_0G['rolling_avg'+str(i)] = ind.rolling(14).mean()
 

#%% 

# time = np.arange(0,len(p53_0G),1)

# for i in range(Ncells):
# # plot using rolling average
#     plt.figure()
#     sns.lineplot( x = time,
#                   y = 'rolling_avg'+str(i),
#                   data = p53_0G,
#                   label = 'Rollingavg')



#%% 

# clustering based on ACF scores
# Max ACF lags
n_lags = 50

doses = [p53_0G, p53_2G, p53_4G, p53_10G ]
doses_norm = []
doses_tr   = []

# # #%% DATA STRUCTURE CORRECTION

for dose in doses:
    doses_norm.append(normalization(np.array(dose)))
    doses_tr.append(np.transpose(normalization(np.array(dose))))

for i in range(4):
    arr =  np.transpose(doses_norm[i]) 
    doses_norm[i] =  arr.tolist() 
    

lag_arr = np.repeat(n_lags, p53_0G.shape[1])
#print(dose.shape[1])

acfval    = list(map(acf, doses_tr[i], lag_arr))
acfval_df = pd.DataFrame(acfval).transpose()

acfval_df.columns = dose.columns
#print(acfval_df.head(20))


plt.figure(figsize=(12, 3))
plt.xlim((1, n_lags))
plt.plot(acfval_df)
plt.show()

