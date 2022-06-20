#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:44:12 2022

@author: nehabinish
"""

"""
Test stationarity of the signals
"""

# --- Import required libraries ---

import numpy as np
import logging
import pandas as pd
import functions
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
#%% INPUT DATA

# --- Load input data into arrays ---
p53_0G  = pd.read_csv('p53_0G.csv' , header=None, delimiter=",")
p53_2G  = pd.read_csv('p53_2G.csv' , header=None, delimiter=",")
p53_4G  = pd.read_csv('p53_4G.csv' , header=None, delimiter=",")
p53_10G = pd.read_csv('p53_10G.csv', header=None, delimiter=",")

#%% CREATE DATABASE OF STATIONARITY

stationarity = {}                             # database(all doses)
alldoses = [p53_0G, p53_2G, p53_4G, p53_10G]
dosename = ['0Gy','2Gy','4Gy','10Gy']    
pscore = []

for (dose,j) in zip(alldoses,range(4)):
   
    Tpoints = dose.shape[0]                   # Time points (120hours/5days)
    Ncells  = dose.shape[1]                   # Cells per dose condition   
    db_list = []                                
    scs = []
    
    for i in range(0,Ncells):
        signal = dose[i]        
        result, score = functions.test_stationarity(signal, 'adfuller')
        scs.append(score)
        
        if(result==True):
            db_list.append('stationary')
        elif(result==False):
            db_list.append('non-stationary')
        else:
            print("Error testing stationarity")
    
    pscore.append(scs)
    stationarity[dosename[j]] = db_list         
    
#%% PERCENTAGE OF NON-STATIONARY SIGNALS

nonstationary_frac = {}
for name in dosename:
    
    test    = stationarity[name]
    nocells = len(test)

    count = 0
    for i in range(nocells):
        if(test[i]=='non-stationary'):
            count += 1
          
    percentage = (count/nocells)*100     
    nonstationary_frac[name] = percentage
    print("Percentage of stationary cells for dose {} is - {}%" .format(name, percentage))                


#%% PLOT FRACTION OF NON_STATIONARITY

plt.figure()
plt.bar(nonstationary_frac.keys(), nonstationary_frac.values())
plt.xlabel('Dose conditions')
plt.ylabel('Percentage of non-stationarity')
plt.ylim(0,100)
plt.axhline(y=50, color='k', linestyle='--', label='50% non-stationarity')
plt.legend()
plt.savefig('st-frac.pdf')
plt.savefig('st-frac.svg')
plt.show()

th=0.05

data = pd.concat(axis=0, ignore_index=True, objs=[ 
    pd.DataFrame.from_dict({'value':pscore[0], 'Dose':'0Gy'}),
    pd.DataFrame.from_dict({'value':pscore[1], 'Dose':'2Gy'}),
    pd.DataFrame.from_dict({'value':pscore[2], 'Dose':'4Gy'}), 
    pd.DataFrame.from_dict({'value':pscore[3],'Dose':'10Gy'})])

fig, ax = plt.subplots()
sns.histplot(
    data=data, x='value', bins = 80, hue='Dose', 
    common_norm=False, ax=ax, alpha=0.4)

plt.axvline(x=th, color='k', linestyle='--', label='threshhold')
plt.xlim(0, 0.5)
plt.xlabel('p-score')
plt.ylabel('Count')
plt.savefig('st-dist.pdf')
plt.savefig('st-dist.svg')













   
