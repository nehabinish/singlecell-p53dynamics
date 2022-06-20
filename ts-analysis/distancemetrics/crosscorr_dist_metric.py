
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:29:43 2022
// population correlations for p53 dynamics (between different doses) - 
// distance metrics for signals //
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
p53_0G  = pd.read_csv('/Users/nehabinish/Desktop/granada internship/data intro analysis/p53_0G.csv'  , header=None, delimiter=",")
p53_2G  = pd.read_csv('/Users/nehabinish/Desktop/granada internship/data intro analysis/p53_2G.csv'  , header=None, delimiter=",")
p53_4G  = pd.read_csv('/Users/nehabinish/Desktop/granada internship/data intro analysis/p53_4G.csv'  , header=None, delimiter=",")
p53_10G = pd.read_csv('/Users/nehabinish/Desktop/granada internship/data intro analysis/p53_10G.csv', header=None, delimiter=",")

#%% INITIALISE DATABASE

alldose  = [p53_2G, p53_4G, p53_10G]
dosename = ['2Gy','4Gy','10Gy',]

#%% EUCLIDEAN DISTANCE COMPARISON OF UNTREATED CELLS WITH DIFFERENT DOSE CONDITIONS
# Euclidean Distance of the signals in the untreated condition with signals 
# from different radiation dose conditions

Eudist = {}
for (dose,k) in zip(alldose, range(3)): 
    print('Dose:' + dosename[k])
    Ncells_dose = dose.shape[1]         # Number of cells given dose cond.
    Ncells_untr = p53_0G.shape[1]       # Number of cells in untreated cond.
    
    val = []
    for untr in range(Ncells_untr):
        sig_untr = p53_0G[untr]
        for tr in range(Ncells_dose):
            print('Untreated Signal ' + str(untr) + ' with ' + dosename[k] + str(tr))
            sig_tr = dose[tr]
            disteu, figure = functions.all_distmetric(sig_untr, sig_tr,
                                              'Euclidean', True)
            val.append(disteu)    
            figure.savefig('C:/Users/binishn/Nextcloud/GranadaLab/Users/Neha/scripts/Eu_distances/'+'untr'+str(untr)+'_' + dosename[k]+ str(tr))
            # signal 1 in figure - untreated cells with specific cell number
            # signal 2 in figure - Damaged cells with specific dose and cell number  
            
    Eudist[dosename[k]] = val           # Database for Euclidean distances   
            
#%% SIGNAL DIFFERENCE BETWEEN DIFFERENT CELLS WITHIN A DOSE CONDITION

Ncells  = p53_0G.shape[1]  
Tpoints = p53_0G.shape[0]  

Eudist_0G = {}
for i in range(0,Ncells-1):
    signal1 = p53_0G[i]
    for j in range(i+1,Ncells):  
        signal2 = p53_0G[j]
        eu_0G, fig0G = functions.all_distmetric(signal1, signal2,
                                          'Euclidean', True) 
        Eudist_0G['signal ' + str(i) + 'with signal ' + str(j)] = eu_0G
        figure.savefig('C:/Users/binishn/Nextcloud/GranadaLab/Users/Neha/scripts/Eu_distances/'
                       +'sig'+str(i)+'_' + 'sig'+ str(j))

#%% CROSS SORRELATION COMPARISON OF TWO SIGNALS

signal1=p53_0G[2]
signal2=p53_4G[10]
plt.plot(signal1)
plt.plot(signal2)
eushifts = functions.all_distmetric(signal1, signal2,
                                  'CrossCorr', True)

#%%






