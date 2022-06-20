#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:25:36 2022
// population correlations for p53 dynamics - distance metrics for signals //
// (within doses) //
@author: nehabinish
"""

#%% IMPORT REQUIRED LIBRARIES

# math libraries
import numpy as np
import logging
import pandas as pd
import random
import statistics

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

#self written modules
import functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%% Input data sets - p53 dynamics

# --- Load input data into arrays ---
p53_0G  = pd.read_csv('p53_0G.csv'  , header=None, delimiter=",")
p53_2G  = pd.read_csv('p53_2G.csv'  , header=None, delimiter=",")
p53_4G  = pd.read_csv('p53_4G.csv'  , header=None, delimiter=",")
p53_10G = pd.read_csv('p53_10G.csv', header=None, delimiter=",")

#%% INITIALISE DATABASE

alldose  = [p53_0G, p53_2G, p53_4G, p53_10G]
dosename = ['0Gy','2Gy','4Gy','10Gy',]

#%% CORRELATION OF SIGNALS WITHIN DOSE CONDITIONS
# correlation of a random signal with all other signals within a dose condition
indexes = []
corr_all = []
for (dose,k) in zip(alldose,range(4)):     
    Ncells  = dose.shape[1]                 # Total cells per dose condition   
    Tpoints = dose.shape[0]                 # Time points (120hours/5day)
    
    corr = []
    for i in range(0,Ncells-1):
        for j in range(i+1,Ncells):  
            indexes.append(dosename[k] + str(i) + str(j))
            print(i,j)
            sig_const = dose[i]
            sig_chng  = dose[j]
            correlates = functions.Corr(sig_const,sig_chng,'same')
            corr.append(correlates)
            
    corr_all.append(corr)    
        
#%% POPULATION CORRELATION TIME-POINT PROPERTIES 
# arranging correlation arrays in terms of time points for averaging 
# time point properties  
  
dose_tp = {}
for j in range(4):
    dose   = np.array(corr_all[j])
    Ncells = dose.shape[0]
    
    tp_prop = []
    for t in range(Tpoints):  
        val=[]
        for no in range(Ncells):          
            case = dose[no]
            val.append(case[t])
        
        tp_prop.append(val)
        
# dose_tp = Different dose conditions - correlation values on the basis
#           of time points for every cell in every dose condition    
    dose_tp[dosename[j]] = tp_prop

#%% AVERAGE TIME_POINT PROPERTIES OF CORRELATION GRAPHS        
   
mean_dose = {}
med_dose  = {}     
for name in dosename:    
    dose = dose_tp[name]
    
    meantp = []
    medtp  = []
    for j in range(len(dose)):
        case = dose[j]
        meantp.append(np.mean(case))
        medtp.append(statistics.median(case))

    mean_dose[name] = meantp
    med_dose[name]  = medtp

#%% PLOT DISTRIBUTION OF AVERAGE TIME-POINT PROPERTIES OF CORRELATION GRAPHS
 # --------------- distribution not relevant here ---------------------------
# Distribution 
# dist = pd.concat(axis=0, ignore_index=True, objs=[ 
#     pd.DataFrame.from_dict({'value':mean_dose['0Gy'],' Dose':'0Gy'}), 
#     pd.DataFrame.from_dict({'value':mean_dose['2Gy'], 'Dose':'2Gy'}),
#     pd.DataFrame.from_dict({'value':mean_dose['4Gy'], 'Dose':'4Gy'}),
#     pd.DataFrame.from_dict({'value':mean_dose['10Gy'],'Dose':'10Gy'})])

# fig, ax = plt.subplots()
# sns.histplot(
#     data=dist,x='value',bins = 100, hue='Dose',stat="density", 
#     common_norm=False, kde=True, ax=ax, alpha=0.4)

# plt.xlabel('Correlation time-averaged properties')
# plt.ylabel('Density of cells')

 # --------------------------------------------------------------------------

#%% PLOT AVERAGE PROPERTIES OF CORRELATION GRAPHS

dt = 0.5

plt.figure()
for name in dosename:
    tvec = np.arange(240)*dt
    plt.plot(tvec, mean_dose[name], label=name)
    plt.ylabel('Mean Correlation area')
    plt.title('Averaged at each time point')
    plt.savefig('C:/Users/binishn/Nextcloud/GranadaLab/Users/Neha/scripts/correlation-distancemetric/'+'mean_'+name)
    plt.legend()
    
plt.figure()
for name in dosename:
    tvec = np.arange(240)*dt
    plt.plot(tvec, med_dose[name], label=name)
    plt.ylabel('Median of Correlation area')
    plt.title('Averaged at each time point')
    plt.savefig('C:/Users/binishn/Nextcloud/GranadaLab/Users/Neha/scripts/correlation-distancemetric/'+'median_'+name)
    plt.legend()
    
 
    
 
    
 
