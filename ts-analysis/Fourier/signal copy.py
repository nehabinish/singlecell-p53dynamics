#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:10:43 2022

@author: nehabinish

"""

#%% IMPORT REQUIRED LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import pyboat.plotting as pl
import logging

from numpy.fft import rfft, rfftfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- To show figures automatically upon creation ---
plt.ion()

#%% FUNCTIONS

def compute_fourier(signal, dt):

    N = len(signal)

    df = 1.0 / (N * dt)  # frequency bins

    rf = rfft(signal, norm="ortho")  # positive frequencies
    # use numpy routine for sampling frequencies
    fft_freqs = rfftfreq(len(signal), d=dt)

    # print(N,dt,df)
    # print(len(fft_freqs),len(rf))

    fpower = np.abs(rf) ** 2 / np.var(signal)  # fourier power spectrum 
    
    meanfp = np.mean(fpower)                   # mean fourier power   
    fpmax  = np.amax(fpower[1:])               # maximum fourier power
    loc = np.where(fpower == fpmax)            # location of the max power
    
    # print(fpmax, meanfp)
    
    loc = loc[0][0] 

    tperiods =  1 / fft_freqs[1:]              # convert freq to periods
    tperiod_fmax = tperiods[loc]               # period for max fourier power
    print(tperiods)

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

#%% SIGNAL ANALYSIS

# --- Choose signal to be analysed ---

signal = p53_0G[5]

dt    = 0.5     # the sampling interval (according to how the data was collected)   - 0.5 hours
lowT  = 2*dt    # lowest period of interest
highT = 120     # highest period of interest

periods = np.linspace(lowT, highT, 500)     # period range, 1hr to 120hr

wAn = WAnalyzer(periods, dt, time_unit_label='hrs')     # initialyse analyser


#%% FILTERING

# --- Sinc Fiter + Detrending + Normalisation ---

# period cut-off for sinc filter
T_c = 70

# calculate the trend with a 50 hr cutoff
trend = wAn.sinc_smooth(signal, T_c) 

# detrending here is then just a subtraction
detrended_signal = signal - trend

# # normalize the amplitude with a sliding window of 70s
# norm_signal = wAn.normalize_amplitude(detrended_signal, window_size=70)

#%% PLOT

# --- Plot raw and filtered signal ---

# plot signal and trend
wAn.plot_signal(signal, label='Raw signal', color='black', alpha=0.5)
#wAn.plot_trend(trend, label='Trend with $T_c$ = {} '.format(T_c))

# --- Plot Normalised signal ---

# # make a new figure to show original signal and detrended + normalized
wAn.plot_signal(signal, num=2, label='Raw signal', color='black', alpha=0.5, marker='.')
# wAn.plot_signal(norm_signal, label='Detrended + normalized', 
#                 alpha=0.8, marker='.')


#%% FOURIR TRANSFORM ON RAW SIGNAL

fourier = compute_fourier(signal, dt)
plot_FFT(signal, dt, fourier[0], fourier[1])


#%%


# --- Plot power versus frequenices ---

# ffq, fpow = compute_fourier(signal,dt)
# fpow = np.log(fpow)
# plt.figure(figsize=(10, 8))
# plt.plot(ffq, fpow)
# plt.xlabel('Frequencies')
# plt.ylabel('Fourier Power')

# plt.figure()
# #plt.plot(p53_0G[:,1])
# plt.plot(freq_p53_0G[:,1])
# plt.xlabel('Time')
# plt.ylabel('Signal')


# ------------------------------------------------------- #


