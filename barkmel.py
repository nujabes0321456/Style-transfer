# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:58:09 2020

@author: USER
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import time 
import math
import wave, array
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from pydub import AudioSegment

def bark_scale(f):
    f=np.array(f)
    bark_list=[]
    for i in range(len(f)):
        bark_list +=[13*math.atan(0.00076*f[i])+3.5*math.atan((f[i]/7500)**2)]
    return bark_list

def dBloss(inputfile,outputfile,diff_spec):
    a = np.asanyarray(inputfile)
    b = np.asanyarray(outputfile)
    b = b*float(diff_spec)
    c=abs(a-b)
    d = abs(a**2/c**2)
    dm = d.mean(axis=0)
    dm = int(dm)
    if dm>0:
        db = 20 * math.log(dm)
        if db>0:
            db_ratio = 1/db
        else:
            db_ratio=1
    else:
        db_ratio=1

    if dm>0:
        print('dB_ratio:',db_ratio,'dB_loss:',db)
    else:
        print('dB_ratio:',db_ratio,'dB_loss:','0')
    

    return db_ratio

def bark_db(inputfile,outputfile,diff_spec):
    dBloss_a=('dBloss_01','dBloss_02','dBloss_03','dBloss_04','dBloss_05','dBloss_06','dBloss_07','dBloss_08','dBloss_09','dBloss_10',
                       'dBloss_11','dBloss_12','dBloss_13','dBloss_14','dBloss_15','dBloss_16','dBloss_17','dBloss_18','dBloss_19',
                       'dBloss_20','dBloss_21','dBloss_22','dBloss_23','dBloss_24')
    dBloss_list=[]
    dBloss_01=dBloss(inputfile[:10],outputfile[:10],diff_spec)
    dBloss_02=dBloss(inputfile[11:20],outputfile[11:20],diff_spec)
    dBloss_03=dBloss(inputfile[21:30],outputfile[21:30],diff_spec)
    dBloss_04=dBloss(inputfile[31:41],outputfile[31:41],diff_spec)
    dBloss_05=dBloss(inputfile[42:53],outputfile[42:53],diff_spec)
    dBloss_06=dBloss(inputfile[54:65],outputfile[54:65],diff_spec)
    dBloss_07=dBloss(inputfile[66:78],outputfile[66:78],diff_spec)
    dBloss_08=dBloss(inputfile[79:92],outputfile[79:92],diff_spec)
    dBloss_09=dBloss(inputfile[93:107],outputfile[93:107],diff_spec)
    dBloss_10=dBloss(inputfile[108:125],outputfile[108:125],diff_spec)
    dBloss_11=dBloss(inputfile[126:145],outputfile[126:145],diff_spec)
    dBloss_12=dBloss(inputfile[146:169],outputfile[146:169],diff_spec)
    dBloss_13=dBloss(inputfile[170:196],outputfile[170:196],diff_spec)
    dBloss_14=dBloss(inputfile[197:230],outputfile[197:230],diff_spec)
    dBloss_15=dBloss(inputfile[231:271],outputfile[231:271],diff_spec)
    dBloss_16=dBloss(inputfile[272:321],outputfile[272:321],diff_spec)
    dBloss_17=dBloss(inputfile[322:382],outputfile[322:382],diff_spec)
    dBloss_18=dBloss(inputfile[383:455],outputfile[383:455],diff_spec)
    dBloss_19=dBloss(inputfile[456:541],outputfile[456:541],diff_spec)
    dBloss_20=dBloss(inputfile[542:641],outputfile[542:641],diff_spec)
    dBloss_21=dBloss(inputfile[642:761],outputfile[642:761],diff_spec)
    dBloss_22=dBloss(inputfile[762:916],outputfile[762:916],diff_spec)
    dBloss_23=dBloss(inputfile[917:1141],outputfile[917:1141],diff_spec)
    dBloss_24=dBloss(inputfile[1142:1542],outputfile[1142:1542],diff_spec)
#    for i,name in enumerate(dBloss_a):
#        dBloss_list += [name]
    
    return dBloss_01,dBloss_02,dBloss_03,dBloss_04,dBloss_05,dBloss_06,dBloss_07,dBloss_08,dBloss_09,dBloss_10,dBloss_11,dBloss_12,dBloss_13,dBloss_14,dBloss_15,dBloss_16,dBloss_17,dBloss_18,dBloss_19,dBloss_20,dBloss_21,dBloss_22,dBloss_23,dBloss_24