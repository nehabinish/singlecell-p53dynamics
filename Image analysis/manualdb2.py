# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:07:13 2022
// Database for manual tracking //
@author: binishn
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import re
import shutil

def split(word):
    return [char for char in word]


#---- Load all images -----

files = []                           # loading only half the time points
path = 'E:/Users/Neha/finalim-tr1/'
valid_images = [".tif"]              # check for valid image types

final_folder= 'E:/Users/Neha/finalim-tr1/'


files = os.listdir(str(path))        # loading only half the time points

#---- Test set images (for one specific well) -----

# for f,i in zip(list_files,range(len(list_files)+1)):
#     if (i%2 == 0):                 # reducing dataset
#         ext = os.path.splitext(f)[1]
#         if os.path.isfile(os.path.join(path,f)) and 'A05_' in f:
#             if ext.lower() not in valid_images:
#                 continue
#             A_05.append(f)


#---- Rename image files -----

# well position
well = ['A05', 'A06', 'B05', 'B06', 'C05', 'C06', 'D05', 'D06']

# positions/site names
site = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12', 's13', 's14', 's15', 's16']

filename = 'E:/Users/Neha/GitHub/imageanalysis/database.txt'
outfile = open(filename, 'w')
for i in range(len(files)):
    
        fname  =  files[i]
        source = path+fname
        splt   =  re.split('_|\.', fname)   
        chno   = split(splt[2])
        glab   = split(splt[5])
        glab   = glab[2] + glab[3] + glab[4]
        print(splt)
        
        fname = splt[5] + 'c' + chno[1] + 't' + splt[4]    
        print(fname)
        outfile.write(fname+'\t'+glab+'\t'+chno[1]+'\t'+ splt[4]+'\n')

outfile.close()
 

        #     files[i] = fname
        #     dest     = final_folder+fname
        
        #     shutil.copyfile(source,dest)


