#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 10 16:49:05 2022

@author: nehabinish
"""

"""
Distribution functions for Fourier time-averaged properties
"""

#%% IMPORT REQUIRED LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import pyboat.plotting as pl
import logging
import pandas as pd
import seaborn as sns

from numpy.fft import rfft, rfftfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%% FUNCTIONS

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def compute_fourier(signal, dt):

    N = len(signal)

    df = 1.0 / (N * dt)  # frequency bins

    rf = rfft(signal, norm="ortho")  # positive frequencies
    # use numpy routine for sampling frequencies
    fft_freqs = rfftfreq(len(signal), d=dt)
    fft_freqs = fft_freqs[12:]

    # print(N,dt,df)
    # print(len(fft_freqs),len(rf))

    fpower = np.abs(rf) ** 2 / np.var(signal)  # fourier power spectrum 
    fpower = fpower[12:]
    
    
    meanfp = np.mean(fpower)                   # mean fourier power   
    closest_mean = find_nearest(fpower, meanfp)
    
    fpmax  = np.amax(fpower)                   # maximum fourier power
    loc = np.where(fpower == fpmax)            # location of the max power
    locmean = np.where(fpower == closest_mean )
    
    # print(fpmax, meanfp)
    
    loc = loc[0][0] 
    locmean = locmean[0][0]

    tperiods =  1 / fft_freqs[1:]              # convert freq to periods
    tperiod_fmax = tperiods[loc]               # period for max fourier power
    tp_mean  = tperiods[locmean-1]
    #print(tperiod_fmax)

    return (np.array([fft_freqs, fpower, meanfp, fpmax, 
                      tperiod_fmax, tp_mean], dtype=object))    
    

def plot_FFT(signal, dt, fft_freqs, fft_power, show_periods=True):
    
    fig = plt.figure(figsize=(10, 8))
    
    ax = pl.mk_Fourier_ax(
        fig, time_unit='hrs', show_periods=show_periods)

    logger.info(f"mean fourier power: {np.mean(fft_power):.2f}")
    pl.Fourier_spec(ax, fft_freqs, fft_power, show_periods)
    fig.tight_layout()
    
#%% INPUT DATA

# --- Load input data into arrays ---
p53_0G = pd.read_csv('p53_0G.csv', header=None, delimiter=",")
p53_2G = pd.read_csv('p53_2G.csv', header=None, delimiter=",")
p53_4G = pd.read_csv('p53_4G.csv', header=None, delimiter=",")
p53_10G = pd.read_csv('p53_10G.csv', header=None, delimiter=",")

Tpoints_0G = p53_0G.shape[0]
Ncells_0G  = p53_0G.shape[1]

Tpoints_2G = p53_2G.shape[0]
Ncells_2G  = p53_2G.shape[1]

Tpoints_4G = p53_4G.shape[0]
Ncells_4G  = p53_4G.shape[1]

Tpoints_10G = p53_10G.shape[0]
Ncells_10G  = p53_10G.shape[1]

#%% INITIALISE LISTS AND ARRAYS

meanpow_0G   = np.zeros(Ncells_0G)
maxpow_0G    = np.zeros(Ncells_0G)
tp_maxpow_0G = np.zeros(Ncells_0G)
tp_mean_0G   = np.zeros(Ncells_0G)

meanpow_2G   = np.zeros(Ncells_2G)
maxpow_2G    = np.zeros(Ncells_2G)
tp_maxpow_2G = np.zeros(Ncells_2G)
tp_mean_2G   = np.zeros(Ncells_2G)

meanpow_4G   = np.zeros(Ncells_4G)
maxpow_4G    = np.zeros(Ncells_4G)
tp_maxpow_4G = np.zeros(Ncells_4G)
tp_mean_4G   = np.zeros(Ncells_4G)

meanpow_10G    = np.zeros(Ncells_10G)
maxpow_10G     = np.zeros(Ncells_10G)
tp_maxpow_10G  = np.zeros(Ncells_10G)
tp_mean_10G    = np.zeros(Ncells_10G)

#%% SIGNAL ANALYSIS

dt    = 0.5     # the sampling interval (according to how the data was collected)   - 0.5 hours
lowT  = 2*dt    # lowest period of interest
highT = 120     # highest period of interest

periods = np.linspace(lowT, highT, 500)     # period range, 1hr to 120hr
wAn = WAnalyzer(periods, dt, time_unit_label='hrs')     # initialyse analyser

for i in range(Ncells_0G):
    
    signal_0G = p53_0G[i]
   
    # --- Sinc Fiter + Detrending + Normalisation ---
    
    # period cut-off for sinc filter
    T_c = 50
    
    # calculate the trend with a 50 hr cutoff
    trend = wAn.sinc_smooth(signal_0G, T_c) 
    
    # detrending here is then just a subtraction
    detrended_signal = signal_0G - trend
    
    # --- Plot signal ---

    # # plot signal and trend 
    # wAn.plot_signal(signal, label='Raw signal', color='red', alpha=0.5)
    # wAn.plot_trend(trend, label='Trend with $T_c$ = {} '.format(T_c)) 
    # plt.savefig('/Users/nehabinish/Desktop/granada internship/scripts/figures/nofilter/signal/cell_{}.pdf'.format(i+1))       # save the figure to file
    # plt.close()  
    
    # --- Compute Fourier Transform ---
    
    fourier_0G = compute_fourier(signal_0G, dt)  
    meanpow_0G[i]   = fourier_0G[2]
    maxpow_0G[i]    = fourier_0G[3]
    tp_maxpow_0G[i] = fourier_0G[4]
    tp_mean_0G[i]   = fourier_0G[5]
    
       
for j in range(Ncells_2G):
    
    signal_2G = p53_2G[j]
    
    # --- Sinc Fiter + Detrending + Normalisation ---
    
    # period cut-off for sinc filter
    T_c = 50
    
    # calculate the trend with a 50 hr cutoff
    trend = wAn.sinc_smooth(signal_0G, T_c) 
    
    # detrending here is then just a subtraction
    detrended_signal = signal_0G - trend
    
    fourier_2G = compute_fourier(signal_2G, dt)  
    meanpow_2G[j]   = fourier_2G[2]
    maxpow_2G[j]    = fourier_2G[3]
    tp_maxpow_2G[j] = fourier_2G[4]
    tp_mean_2G[j]   = fourier_2G[5]


for k in range(Ncells_4G):  

    signal_4G = p53_4G[k]
    
    # --- Sinc Fiter + Detrending + Normalisation ---
    
    # period cut-off for sinc filter
    T_c = 50
    
    # calculate the trend with a 50 hr cutoff
    trend = wAn.sinc_smooth(signal_0G, T_c) 
    
    # detrending here is then just a subtraction
    detrended_signal = signal_0G - trend
    
    fourier_4G = compute_fourier(signal_4G, dt)   
    meanpow_4G[k]   = fourier_4G[2]
    maxpow_4G[k]    = fourier_4G[3]
    tp_maxpow_4G[k] = fourier_4G[4]
    tp_mean_4G[k]   = fourier_4G[5]
    
for l in range(Ncells_10G):  

    signal_10G = p53_10G[l]
    
    # --- Sinc Fiter + Detrending + Normalisation ---
    
    # period cut-off for sinc filter
    T_c = 50
    
    # # calculate the trend with a 50 hr cutoff
    # trend = wAn.sinc_smooth(signal_0G, T_c) 
    
    # # detrending here is then just a subtraction
    # detrended_signal = signal_0G - trend
    
    fourier_10G = compute_fourier(signal_10G, dt)   
    meanpow_10G[l]   = fourier_10G[2]
    maxpow_10G[l]    = fourier_10G[3]
    tp_maxpow_10G[l] = fourier_10G[4]
    tp_mean_10G[l]   = fourier_10G[5]
    


#%%  PLOT TP - MAX
  
#plt.hist(meanpow_0G, bins=10)
# seaborn histogram

# plt.figure()
# plt.hist([tp_maxpow_0G, tp_maxpow_2G, tp_maxpow_4G], color=['k','b','g'], alpha=0.5)
# plt.show()

df = pd.concat(axis=0, ignore_index=True, objs=[ 
    pd.DataFrame.from_dict({'value': tp_maxpow_0G, 'Dose': '0G'}), 
    pd.DataFrame.from_dict({'value': tp_maxpow_2G, 'Dose': '2G'}), 
    pd.DataFrame.from_dict({'value': tp_maxpow_4G, 'Dose': '4G'}),
    pd.DataFrame.from_dict({'value': tp_maxpow_10G, 'Dose': '10G'})])

fig, ax = plt.subplots()
sns.histplot(
    data=df, x='value', bins = 50, hue='Dose', stat="density", common_norm=False, kde=True, ax=ax, alpha=0.4)
plt.xlabel('Time period - maximum fourier power')
plt.ylabel('Density of cells')

# fig, ax = plt.subplots()
# for a in [tp_maxpow_0G, tp_maxpow_2G, tp_maxpow_4G]:
#     sns.distplot(a, ax=ax, kde=True)
# plt.xlabel('Time period - maximum fourier power')
# plt.ylabel('Density of cells')

#%% PLOT - MEAN TIMEPERIOD

  
#plt.hist(meanpow_0G, bins=10)
# seaborn histogram

# plt.figure()
# plt.hist([tp_maxpow_0G, tp_maxpow_2G, tp_maxpow_4G], color=['k','b','g'], alpha=0.5)
# plt.show()

df = pd.concat(axis=0, ignore_index=True, objs=[ 
    pd.DataFrame.from_dict({'value': tp_mean_0G, 'Dose': '0G'}), 
    pd.DataFrame.from_dict({'value': tp_mean_2G, 'Dose': '2G'}), 
    pd.DataFrame.from_dict({'value': tp_mean_4G, 'Dose': '4G'}),
    pd.DataFrame.from_dict({'value': tp_mean_10G, 'Dose': '10G'})])

fig, ax = plt.subplots()
sns.histplot(
    data=df, x='value', bins = 20, hue='Dose', stat="density", common_norm=False, kde=True, ax=ax, alpha=0.4)
plt.xlabel('Charecterstic Time period ')
plt.ylabel('Density of cells')

# fig, ax = plt.subplots()
# for a in [tp_maxpow_0G, tp_maxpow_2G, tp_maxpow_4G]:
#     sns.distplot(a, ax=ax, kde=True)
# plt.xlabel('Time period - maximum fourier power')
# plt.ylabel('Density of cells')

#%% PLOT - MEAN
ds = pd.concat(axis=0, ignore_index=True, objs=[ 
    pd.DataFrame.from_dict({'value': meanpow_0G, 'Dose': '0G'}), 
    pd.DataFrame.from_dict({'value': meanpow_2G, 'Dose': '2G'}), 
    pd.DataFrame.from_dict({'value': meanpow_4G, 'Dose': '4G'}),
    pd.DataFrame.from_dict({'value': meanpow_10G, 'Dose': '10G'})])

fig, ax = plt.subplots()
sns.histplot(
    data=ds, x='value', bins = 100, hue='Dose', stat="density", 
    common_norm=False , kde=True, ax=ax, alpha=0.4)
plt.xlabel('Mean Fourier Power')
plt.ylabel('Density of cells')

#%% PLOT - MAX

dmax = pd.concat(axis=0, ignore_index=True, objs=[ 
    pd.DataFrame.from_dict({'value': maxpow_0G, 'Dose': '0G'}), 
    pd.DataFrame.from_dict({'value': maxpow_2G, 'Dose': '2G'}), 
    pd.DataFrame.from_dict({'value': maxpow_4G, 'Dose': '4G'}),
    pd.DataFrame.from_dict({'value': maxpow_10G, 'Dose': '10G'})])

fig, ax = plt.subplots()
sns.histplot(
    data=dmax, x='value', bins = 50, hue='Dose', stat="density", 
    common_norm=False, kde=True, ax=ax, alpha=0.4)
plt.xlabel('Energy')
plt.ylabel('Density of cells')

# fig, ax = plt.subplots()
# for a in [meanpow_0G, meanpow_2G, meanpow_4G]:
#     sns.distplot(a, ax=ax, kde=True)
# plt.xlabel('Mean Fourier Power')
# plt.ylabel('Density of cells')


# #%%

# f, ax = plt.subplots()
# ax.bar(x=p53_0G[0],
#        height=p53_0G[0])

# plt.figure()
# for i in range(Ncells_0G):
#     # seaborn histogram
# #     
# sns.distplot(p53_0G[0], hist=True, kde=True, 
#                   bins=int(240/10), color = 'blue',
#                   hist_kws={'edgecolor':'black'})


#%%

