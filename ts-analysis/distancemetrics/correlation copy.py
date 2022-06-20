#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:17:17 2022
// cross and auto correlation of two signals exhibiting p53 dynamics //
@author: nehabinish
"""

#%% IMPORT REQUIRED LIBRARIES

# math libraries
import numpy as np
import logging
import pandas as pd
from scipy.integrate import simps
from scipy.signal import correlate
import math

# plotting libraries
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
#%% Input data sets - p53 dynamics

# --- Load input data into arrays ---
p53_0G  = pd.read_csv('p53_0G.csv'  , header=None, delimiter=",")
p53_2G  = pd.read_csv('p53_2G.csv'  , header=None, delimiter=",")
p53_4G  = pd.read_csv('p53_4G.csv'  , header=None, delimiter=",")
p53_10G = pd.read_csv('p53_10G.csv', header=None, delimiter=",")

Tpoints_0G = p53_0G.shape[0]
Ncells_0G  = p53_0G.shape[1]

Tpoints_2G = p53_2G.shape[0]
Ncells_2G  = p53_2G.shape[1]

Tpoints_4G = p53_4G.shape[0]
Ncells_4G  = p53_4G.shape[1]

#%% Function for correlation distance metric

def dist_corr(sig1, sig2):
    
    # check if padding is required
    if (len(sig1) == len(sig2)):
        print("padding not required")
      
    else:
        print("padding signals ...")
        padding_length = len(sig1) - len(sig2)
        
        # check which signal should be padded
        if(len(sig1) < len(sig2)):
            np.pad(sig1, (math.floor(padding_length/2.0), 
                                                math.ceil(padding_length/2.0)), 
                                        mode = 'constant', constant_values=0)     
        elif(len(sig2) < len(sig1)):
            np.pad(sig2, (math.floor(padding_length/2.0), 
                                                math.ceil(padding_length/2.0)), 
                                        mode = 'constant', constant_values=0)
            
        
    correlate_result = correlate(sig1, sig2, mode='same', method='auto')
    shift_positions  = np.arange(-len(sig1) + 1, len(sig2))
    print(shift_positions) # The shift positions of b
    
    return correlate_result

def area(signal):
    
    # trapezoidal method  
    area_trapz = np.trapz(signal, x=None, dx=0.5)
    # Simpson's rule.
    area_sims  = simps(signal, x=None, dx=0.5)
    
    return area_trapz, area_sims
        
#%% Initialise signals

# choose signals        
signal1 = p53_4G[1]
signal2 = p53_4G[2]

sig3 = p53_4G[1]
sig4 = p53_4G[3]

dt   = 0.5     # the sampling interval (according to how the data was collected)   - 0.5 hours
tvec = np.arange(len(signal1)) * dt # total time in hours


corr_dist  = dist_corr(signal1, signal2)
corr_dist2 = dist_corr(sig3, sig4)

# plot signals and correlations
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,8))
fig.suptitle('Signals')
ax1.plot(tvec,signal1,  'k-')
ax1.plot(tvec,signal1,  'r.')
ax1.title.set_text('Signal 1')
ax2.plot(tvec,signal2,  'r-')
ax2.plot(tvec,signal2,  'k.')
ax2.title.set_text('Signal 2')
ax3.plot(corr_dist)  

#%% Correlation distance metric 

corr_dist = dist_corr(signal1, signal2)
plt.figure(figsize=(10,8))
plt.plot(corr_dist)     
plt.xlabel('Time points') 
plt.ylabel('Correlated area')

corr_area = area(corr_dist)
print(corr_area)

#%%

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,8))
fig.suptitle('Signals')
ax1.plot(tvec,sig3,  'k-')
ax1.plot(tvec,sig3,  'r.')
ax1.title.set_text('Signal 1')
ax2.plot(tvec,sig4,  'r-')
ax2.plot(tvec,sig4,  'k.')
ax2.title.set_text('Signal 2')
ax3.plot(corr_dist)     
    
        
#%%
    
    
    
    
    
    
    
    
    
