#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:49:04 2022

@author: nehabinish
"""

"""
Fourier Ensemble properties
"""

#%% IMPORT REQUIRED LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import pyboat.plotting as pl
import logging
plt.rcParams.update({'font.size': 15})

from numpy.fft import rfft, rfftfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#%% FUNCTIONS

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
    fpmax  = np.amax(fpower)                   # maximum fourier power
    loc = np.where(fpower == fpmax)            # location of the max power
    
    # print(fpmax, meanfp)
    
    loc = loc[0][0] 

    tperiods =  1 / fft_freqs[1:]              # convert freq to periods
    tperiod_fmax = tperiods[loc]               # period for max fourier power
    #print(tperiod_fmax)

    return (np.array([fft_freqs, fpower, meanfp, fpmax, 
                      tperiod_fmax], dtype=object))    
    

def plot_FFT(signal, dt, fft_freqs, fft_power, show_periods=True):
    
    fig = plt.figure(figsize=(10, 8))
    
    ax = pl.mk_Fourier_ax(
        fig, time_unit='hrs', show_periods=show_periods)

    logger.info(f"mean fourier power: {np.mean(fft_power):.2f}")
    pl.Fourier_spec(ax, fft_freqs, fft_power, show_periods)
    fig.tight_layout()
    

#%% INPUT DATA

# --- Load input data into arrays ---
p53_0G = np.loadtxt('p53_0G.csv', delimiter=",")
p53_2G = np.loadtxt('p53_2G.csv', delimiter=",")
p53_4G = np.loadtxt('p53_4G.csv', delimiter=",")

Tpoints = p53_4G.shape[0]
Ncells  = p53_4G.shape[1]


#%% INITIALISE LISTS AND ARRAYS

meanpow   = np.empty(Ncells)
maxpow    = np.empty(Ncells)
tp_maxpow = np.empty(Ncells)


#%% SIGNAL ANALYSIS

dt    = 0.5     # the sampling interval (according to how the data was collected)   - 0.5 hours
lowT  = 2*dt    # lowest period of interest
highT = 120     # highest period of interest
    
periods = np.linspace(lowT, highT, 500)     # period range, 1hr to 120hr
wAn = WAnalyzer(periods, dt, time_unit_label='hrs')     # initialyse analyser

for i in range(Ncells):
    
    signal = p53_4G[:,i]
    
    # --- Sinc Fiter + Detrending + Normalisation ---
    
    # period cut-off for sinc filter
    T_c = 50
    
    # calculate the trend with a 50 hr cutoff
    trend = wAn.sinc_smooth(signal, T_c) 
    
    # detrending here is then just a subtraction
    detrended_signal = signal - trend
    
    # --- Plot signal ---

    # # plot signal and trend 
    # wAn.plot_signal(signal, label='Raw signal', color='red', alpha=0.5)
    # wAn.plot_trend(trend, label='Trend with $T_c$ = {} '.format(T_c)) 
    # plt.savefig()       # save the figure to file
    # plt.close()  
    
    # --- Compute Fourier Transform ---
    
    fourier = compute_fourier(signal, dt)
    
    meanpow[i]   = fourier[2]
    maxpow[i]    = fourier[3]
    tp_maxpow[i] = fourier[4]
    
    
    # # --- Plot Fourier Power Spectra ---
    
    # # plot fourier transform of the signal
    # plot_FFT(signal, dt, fourier[0], fourier[1])
    # plt.savefig()   # save the figure to file
    # plt.close() 
    
    
#%% PLOT ENSEMBLE VALUES

cells = np.arange(1,Ncells+1,1)  

# --- Mean Power Spectra for all cells ---  
    
pop_meanfp = np.mean(meanpow) 

plt.figure(figsize=(6, 4))   
plt.plot(cells, meanpow)
plt.axhline(y=pop_meanfp, color='r', linestyle='--')
plt.xlabel('Cells') 
plt.ylabel('Mean Fourier Power')
plt.savefig('meanfp.pdf')

# --- Max Power Spectra for all cells ---  

plt.figure(figsize=(10, 4))   
for i in range(Ncells):
    
    point1 = [cells[i], 0]
    point2 = [cells[i], maxpow[i]]
    
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    
    plt.plot(x_values, y_values)
    
    plt.xlabel('Cells') 
    plt.ylabel('Maximum Fourier Power')

plt.savefig('maxfp.pdf')

# --- Time Period of Max Power Spectra for all cells ---  
    
pop_tmaxfp = np.mean(tp_maxpow) 
print('Mean Time period coressponding to Maximum Fourier Power - {}' .format(pop_tmaxfp))

plt.figure(figsize=(6, 4))   
plt.plot(cells, tp_maxpow, '.')
plt.axhline(y=pop_tmaxfp, color='r', linestyle='--')
plt.ylim(0,15)
plt.xlabel('Cells') 
plt.ylabel('Time period for Maximun Fourier Power')
plt.savefig('maxtp.pdf')
# ------------------------------------------------------- #

