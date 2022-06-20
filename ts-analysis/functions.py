#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:01:25 2022
// functions for p53 dynamics analysis //
@author: nehabinish

"""

"""
========================================================================
        Functions for single-cell time series analysis
========================================================================
"""

# --- Import required libraries ---
import numpy as np
from pyboat import WAnalyzer
import logging
import pandas as pd
from scipy.integrate import simps
from scipy.signal import correlate
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# --- Professor Jon E. Froehlich open source library ---
import utility


# --- Differential equations Integrator  ---
def RK4(f, x0, t0, t1, dt):
    """
    ========================================================================    
                Integrator - RK4 iterative method
            
    Arguments:
        f(function): Integrable Differential equations
        x0(one dimensional array): Initial values
        t0(float): Initial time point
        t1(float): Final time point
        dt(float): Integration step size 
    Return:
        X(one dimensional array): [dx0/dt] - Integrated output
    ========================================================================
    """
    
    t = np.arange(t0, t1, dt)   #time span 
    N = len(t)
    X = np.empty((len(t), len(x0)))
    X[0] = x0
    
    for i in range(1, N):
        
        k1 = f(X[i-1], t[i-1])
        k2 = f(X[i-1] + dt/2*k1, t[i-1] + dt/2)
        k3 = f(X[i-1] + dt/2*k2, t[i-1] + dt/2)
        k4 = f(X[i-1] + dt*k3, t[i-1] + dt)
        
        X[i] = X[i-1] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
    return X           


def Pad_signals(sig1, sig2):
    """
    ========================================================================    
                Pad smaller signals with zeroes on both sides 
            
    Arguments:
        sig1(one dimensional array): First input signal
        sig2(one dimensional array): Second input signal 
    Return:
        sig1, sig2(one dimensional arrays): Padded signals
    ========================================================================
    """

    print("padding signals ...")
    padding_length = len(sig1) - len(sig2)
    
    # check which signal should be padded
    if(len(sig1)<len(sig2)):
        np.pad(sig1, (math.floor(padding_length/2.0), 
                                            math.ceil(padding_length/2.0)), 
                                    mode='constant', constant_values=0)
    elif(len(sig2)<len(sig1)):
        np.pad(sig2, (math.floor(padding_length/2.0), 
                                            math.ceil(padding_length/2.0)), 
                                   mode='constant', constant_values=0)
        
    return sig1, sig2    


# --- Correlation of signals ---
def Corr(sig1, sig2, mode):   
    """
    ========================================================================    
            Find the correlation result of two input signals
            
    Arguments:
        sig1(one dimensional array): First input signal
        sig2(one dimensional array): Second input signal 
        mode(string{‘full’,‘same’}):  Correlation modes
            'full' - full discrete linear cross-correlation
            'same' - same size as inputs, centered with respect to ‘full’ 
    Return:
        correlate_result(one dimensional array): Correlation values
    ========================================================================
    """
    
    # check if padding is required
    if(len(sig1)==len(sig2)):
        print("padding not required")     
    else:
        sig1, sig2 = Pad_signals(sig1, sig2)
                    
    correlate_result = correlate(sig1, sig2, mode='same', method='auto')
    shift_positions  = np.arange(-len(sig1) + 1, len(sig2))
    #print(shift_positions) # The shift positions of b

    return correlate_result


# --- Area under the signal curves ---
def area(signal, mode):
    """
    ========================================================================    
                    Find the area under the signal
            
    Arguments:
        signal(one dimensional array): Input signal
        mode(string{‘trapz’,‘sims’}): Method of integration 
            'trapz' - Integration using composite trapezoidal rule
            'sims' - Integration using composite simpson rule
    Return:
        area(float): Area under the curve
    ========================================================================
    """
    
    if(mode=='trapz'):
        # trapezoidal method  
        area = np.trapz(signal, x=None, dx=0.5)
    elif(mode=='simps'):
        # Simpson's rule.
        area = simps(signal, x=None, dx=0.5)
        
    return area


# --- Correlation area comparison of two signals ---
def dist_corr(set1, set2):
    """
    ========================================================================    
        Distance metric employed using area under correlation graph
            
    Arguments:
        set1(two dimensional array): First set of signal to be compared
        set2(two dimensional array): Second set of signals to be compared
    Return:
        dom_signal(two dimensional array): More correlated set of signal
    ========================================================================
    """

    corr_area1 = Corr(set1[0], set1[1], mode='same')
    corr_area2 = Corr(set2[0], set2[1], mode='same')

    if(corr_area1>corr_area2):      
        print('Signals in set 1 is more correlated')
        dom_signal = set1
    elif(corr_area1<corr_area2):
        print('Signals in set 2 is more correlated')
        dom_signal = set2

    return dom_signal


# --- Testing stationarity of signal ---
def test_stationarity(signal, mode, scale='norm', eps=10**(-2), th=0.05):
    """
    ========================================================================    
        Test stationarity of a signal employing methods input using mode
            
    Arguments:
        signal(one dimensional array): Input Signal 
        mode(string{‘summary’,‘adfuller}): Method used
            'summary'  - Compare mean and variance of the split dataset
                         (default) tolerence = 10^(-2)
            'adfuller' - Unit root statistical test method specifically, 
                         Augmented Dickey-Fuller test 
                         (default) threshhold = 0.05
        scale(string{‘norm’,‘log’})): Input scale of dataset  
            'norm' - Conform to dataset scale      (default)
            'log'  - Test on log scale of dataset
    Return:
        result(Boolean): 'True' if signal is stationary, 'False' if not
    ========================================================================
    """
    
    # calculate mean and variance of the data
    mean_data = np.mean(signal)
    var_data  = np.var(signal)
    print("Mean of dataset - {} \nVariance of dataset - {}\n" 
          .format(mean_data, var_data))

    # method -
    
    # 1. summary statistics of the dataset
    if(mode=='summary'):
        # compare first and second half of signal to check stationarity of signal
        # compare mean and variance fluctuations
        
        length = len(signal)        
        if(length%2==0):
            splt_len = int(length/2)
        elif(length%2!=0):
            splt_len = int((length-1)/2)
            splt_len = (length-1)/2.0
        
        fhalf = signal[0:splt_len]
        shalf = signal[splt_len:]
        
        mean1, mean2 = fhalf.mean(), shalf.mean()
        print("Mean of first half - {} \nMean of second half - {}\n" 
              .format(mean1, mean2))
        var1, var2   = fhalf.var(), shalf.var()
        print("Variance of first half - {} \nVariance of second half - {}\n" 
              .format(var1, var2))
        
        # log components
        signal_log = np.log(signal)
        fhalf_log  = signal_log[0:splt_len]
        shalf_log  = signal[splt_len:]
        mean1_log, mean2_log = fhalf_log.mean(), shalf_log.mean()
        var1_log, var2_log   = fhalf_log.var(), shalf_log.var()

        if(np.abs(mean1-mean2)/100<eps and 
            np.abs(var1-var2)/100<eps):
            print("comparing data ...")
            # check tolerance limits of fluctuations
            print("The signal is stationary")
            result = True            
        elif(np.abs(mean1_log-mean2_log)/100<eps and
             np.abs(var1_log-var2_log)/100<eps):
            print("comparing log components of data ...")
            # check tolerance limits of fluctuations - logscale
            print("The signal is stationary")
            result = True   
        else:
            print("The signal is non stationary")
            result = False    
    
    # 2. Augmented Dickey-Fuller test
    if(mode=='adfuller'):
        # null hypothesis (stationary), 
        # if p-value above threshold - 
        # fail to reject null hypothesis(non-stationary).
        
        adf_signal = adfuller(signal)
        # convert to log scale
        if(scale=='log'):
            adf_signal = adfuller(np.log(signal))
        
        adf_st = adf_signal[0]
        pval   = adf_signal[1]
    
        ## --- uncomment to print return values of adfuller test ---
        # print('p-value: {}' .format(pval))
        # print('ADF Statistic: {}' .format(adf_st))
        # print('Critical Values:')
        # for key, value in adf_signal[4].items():
        #  	print('\t%s: %.3f' % (key, value))
        
        if(pval>th):
            print("Hypothesis fails to be rejected, signal is non stationary")
            result = False    
        elif(pval<=th)  :
            print("Reject null hypothesis, signal is stationary")
            result = True
        elif(pval==0.0):
            print("Error: p value = 0")
        else:
            print("Unknown error occured")
            
        # Compare ADF statistics with critical values
        # Determine significance level for rejection
        critical_vals = adf_signal[4]
        if(adf_st<critical_vals['1%']):
            print(
            'Hypothesis rejected with a significance level of less than 1%')
        elif(adf_st<critical_vals['5%']):
            print(
            'Hypothesis rejected with a significance level of less than 5%')
        elif(adf_st<critical_vals['10%']):
            print(
            'Hypothesis rejected with a significance level of less than 10%')      
               
    return result, pval


# --- Correlation between two different dose conditions ---    
def dose_corr(dose1, dose2):
    """
    ========================================================================    
    Given two dose conditions, find correlation between p53 signals of cells
            
    Arguments:
        dose1(two dimensional array): p53 signals from all cells within a dose 
                                      condition
        dose2(two dimensional array): p53 signals from all cells within 
                                      another dose condition  
    Return:
        corr_all(array) : correlation area of all signals in one dose 
                          condition with all signals of another dose
    ========================================================================
    """
    indexes,corr_all,corr = []
    Ncells_dose1 =  dose1.shape[0]       # Number of cells in first dose cond
    Ncells_dose2 =  dose2.shape[0]       # Number of cells in second dose cond
    for i in range(Ncells_dose1):
        d1sig = dose1[i]
        for j in range(Ncells_dose2):  
            indexes.append(str(i) + str(j))
            d2sig = dose2[j]
            correlates = Corr(d1sig, d2sig,'same')
            corr.append(correlates)
                
        corr_all.append(corr)    
    return corr_all


# --- Distance metric for two signals ---  
def all_distmetric(sig1, sig2, mode, plot):      
    """
    ========================================================================    
    Different modes applied to quantify differences and similarities between
                                two input ignals            
    Arguments:
        sig1(one dimensional array): First input signal
        sig2(one dimensional array): Second input signal  
        mode(string{‘Euclidean’): Method used
            'Euclidean' - The Euclidean distance between each point of the 
                           signal
            'CrossCorr' - Cross Correlate two signals and shift the second
                          signal to have less diffrence in euclidean distance 
       plot(boolean): 'True' for plot generation, 'False' otherwise                    
    Return:
        result(float/array): Quantized results 
    ========================================================================
    """
      
    # check if padding is required
    if(len(sig1)==len(sig2)):
        print("padding not required")     
    else:
        sig1, sig2 = Pad_signals(sig1, sig2)
    
    if(mode=='Euclidean'):
        print("Calculating Euclidean Distance ... ")
        eu_dist = utility.distance.euclidean(sig1, sig2)
        result  = eu_dist
        
        if(plot==True):
            dt    = 0.5                      # sampling rate
            tvec  = np.arange(240)*dt
            fig, axes = plt.subplots(1, 1, figsize=(12, 4))
            axes.plot(tvec, sig1, alpha=0.7, label="signal 1", marker="o")
            axes.plot(tvec, sig2, alpha=0.7, label="signal 2", marker="D")
    
            # draw connecting segments between a_i and b_i used for Euclidean 
            # distance calculation
            axes.vlines(np.arange(0, 240*dt, dt), sig1, sig2, alpha = 0.7)
            axes.legend()
            axes.set_title("Raw signals | Euclidean distance = {:0.1f}"
                           .format(eu_dist))
            axes.set_xlabel("Time (in hours)")
            
    if(mode=='CrossCorr'):
        print("Cross Correlating signals ...")
        eudist_aftershift = utility.compare_and_plot_signals_with_alignment(
            sig1, sig2, bshift_method = 'all')
        result =  eudist_aftershift
        
    return result                
        
        
              
        

# new function for plots in files




