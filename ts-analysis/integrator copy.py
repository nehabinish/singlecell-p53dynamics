#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:20:14 2021

@author: nehabinish
"""

import numpy as np
import math 


#DEFINING THE RUNGE KUTTA 4 INTEGRATOR
def RK4(f, x0, t0, t1, dt):
    
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

