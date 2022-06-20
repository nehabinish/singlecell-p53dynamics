# -*- coding: utf-8 -*-
"""
Image analysis
04/03/2022 - Start of development date 

author - Neha Binish

"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import re
import shutil

#---- Load all images -----

files = []                           # loading only half the time points
path = 'E:/Users/Neha/im-tr1/'
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

for i in range(len(files)):
    
        fname  =  files[i]
        source = path+fname
        splt   =  re.split('_|\.', fname)   
        print(splt)
 
        if(int(splt[4])%2 == 0):
        
            # rename time points
            ln = len(splt[4]) 
            if (ln == 1):        
                splt[4] = '00' + splt[4]
            elif (ln == 2):
                splt[4] = '0' + splt[4]
            
            # fname = "_".join(splt[:5]) + '.' + splt[5]
        
            # check for well position
            if splt[0] not in well:
                print('error - well index not found')
            else:    
                wellpos = well.index(splt[0])
            
            # check for site position
            if splt[1] not in site:
                print('error - site index not found')
            else:        
                gname = splt[1].replace('s', '0')
            
            # add group names 
            if (wellpos != 0):
                gname = str(int(gname) + (wellpos * 16))
                
                # add zeros to group name
                ln = len(gname) 
                if (ln == 1):        
                    gname = '00' + gname
                elif (ln == 2):
                    gname = '0'  + gname     
                
            fname = "_".join(splt[:5]) + '_xy' + gname + '.' + splt[5]    
            print(fname)
            
            files[i] = fname
            dest     = final_folder+fname
        
            shutil.copyfile(source,dest)


