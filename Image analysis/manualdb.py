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


# Opening and Closing a file "MyFile.txt"
# for object name file1.
f = open("E:/Users/Neha/GitHub/imageanalysis/database.txt","w")
f.write('filename \t group_label \t position_number \t channel_name \t timepoint\n')
for i in range(len(files)):
    
        fname  =  files[i]
        source = path+fname
        splt   =  re.split('_|\.', fname)   
        print(splt)
        chno  = [ch for ch in splt[2]]   
        posno = [ch for ch in splt[5]]
        posno = posno[2]+ posno[3]+ posno[4]
        fname = splt[5]+'c'+chno[1]+'t'+splt[4]+'.tif'
        
        f.write(fname+' \t '+' '+' \t '+posno+' \t '+chno[1]+' \t '+splt[4]+'\n')
        print(fname)
        
f.close()       
            
        #     files[i] = fname
        #     dest     = final_folder+fname
        
        #     shutil.copyfile(source,dest)


