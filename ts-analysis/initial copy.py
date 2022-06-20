#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:10:41 2022

@author: nehabinish
"""

# Import Libraries
import math 
import numpy as np

# Parameters and constants
a = 30.5                # maximal self activation rate of ATM
p = 22                  # rate of dephosphorylation of ATM by Wip1
c = 1.4                 # production rate of p53
g = 2.5                 # maximal degradation of P53 by Mdm2

trm = 1                 # maximal production rate of mdm2
tpm = 4                 # maximal production rate of Mdm2
trw = 1                 # maximal production rate of wip1
tpw = 1                 # maximal production rate of Wip1

dam = 20                # degradation rate of Mdm2 by ATM*
da  = 0.16              # basal dephosphorylation rate of ATM*
dp  = 0.1               # basal degradation rate of P53
drm = 1                 # basal degradation rate of mdm2
dpm = 2                 # basal degradation rate of Mdm2
drw = 1.3               # basal degradation rate of wip1
dpw = 2.3               # basal degradation rate of Wip1

ka  = 0.5               # Michaelis constant for the ATM* self activation
kwa = 0.14              # Michaelis constant for the inhibition of the ATM* self activation by Wip1
kmp = 0.15              # Michaelis constant for the degradation of P53 by Mdm2
kpm = 1                 # Michaelis constant for the production of mdm2 by P53 
kpw = 1                 # Michaelis constant for the production of wip1 by P53

r = 2                   # strength of the inhibition of the Mdm2 mediated degradation of P53 by ATM*
k = 1/20                # which value?

smax  = 0.2             # maximal signal strength of the DSB process
gamma = 9               # Michaelis contant for the signal strength of the DSB process


# Integration constants
L  = 2                 # Length  in space
t0 = 0                 # Initial integration time
tf = 120               # Final integration time in hours
dx = 0.1               # Space step size 
dt = 0.01              # Time step size

# sampling/acquisition rate
sampling = 1

# DSB white noise initial constants
# mean    = 1
# std     = 1

mean    = 20
std     = 5
eps     = 0.002
intens  = 1
dsb     = 0
ini_DSB = 1
timethresh = 200

