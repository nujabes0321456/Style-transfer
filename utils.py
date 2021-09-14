# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:52:39 2019

@author: USER
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import time 
import math
import wave, array
from barkmel import bark_scale as bks
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from pydub import AudioSegment

AudioSegment.converter ='D:\\ffmpeg-20191217-bd83191-win64-static\\ffmpeg-20191217-bd83191-win64-static\\bin\\ffmpeg.exe'
N_FFT = 4096


def read_audio_spectum(filename):
    x, fs = librosa.load(filename,44100)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S[:,:800])
    S = np.log1p(np.abs(S[:,:800]))  
    return S, fs ,p

def freq_domin_amp(filename):
#    x, fs = librosa.load(filename)
    fs, x = wav.read(filename)
    if fs==44100 :
        T = 1/fs
        t = 0.1 
        N = fs*t
        omega=np.angle(x)
        Y_k = np.fft.fft(x)[0:int(N/2)]/N # FFT function from numpy
        Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
        Pxx = np.abs(Y_k) # be sure to get rid of imaginary part
        f = fs*np.arange((N/2))/N; # frequency vector
        bak_f=bks(f);
        fig,ax = plt.subplots()
        plt.plot(f,Pxx[:,0],linewidth=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency [Hz]')
        plt.show()
        print(sum(abs(Pxx)))      
    else :
        print('error please check fs = 44100Hz')
    return f, Pxx[:,0],bak_f

def signaltonoise(a, axis, ddof): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 


def make_stereo(x):
    x1=np.zeros(1)
    x1=x
    x1 = np.asanyarray(x1)
    x2=np.zeros((2,len(x1)),dtype='float32')
    x2[0,:]=x1*0.5
    x2[1,:]=x1*0.5
    x2 = np.asanyarray(x2)
    x2 = np.asfortranarray(x2)
    return x1, x2


def vol_balence(content,output):
    song_1 = AudioSegment.from_wav(content)
    song_2 = AudioSegment.from_wav(output)
    o_S=str(output)
    samples_1 = song_1.get_array_of_samples()
    samples_1 = np.array(samples_1)
    samples_2 = song_2.get_array_of_samples()
    samples_2 = np.array(samples_2,dtype='int16')
#    if np.max(np.abs(samples_2))>2000:
#        v_balance=np.mean(np.abs(samples_1))/(np.mean(np.abs(samples_2))/50)
#    else:
#        v_balance=np.mean(np.abs(samples_1))/(np.mean(np.abs(samples_2)))
    blan_file=song_2+25
#    samples_3 = blan_file.get_array_of_samples()
#    samples_3 = np.array(samples_3)
    blan_file.export("audio_data/balance/"+o_S,format='wav')
    return

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    h -= h * 60
    return '%dh %dm %ds' % ( h, m, s)




    
#test data  
#arr1=[]
#def signaltonoise(a, axis=0, ddof=0):
#    arr1 = np.array(a_content)
#    arr1 = arr1.tolist()
#    arr1 =[x for j in arr1 for x in j]
#    arr2 = np.array(a)
#    arr2= arr2.tolist()
#    arr2 =[x for j in arr2 for x in j]
#    a=[arr1,arr2]
#    a = np.asanyarray(a) 
#    m = a.mean(axis) 
#    sd = a.std(axis = axis, ddof = ddof) 
#    return np.where(sd == 0, 0, m / sd) 
#  
#print ("\narr1 : ", arr1) 
#print ("\narr2 : ", arr2) 
#  
##print ("\nsignaltonoise ratio for arr1 : ",  
##       signaltonoise(arr1, axis = 0, ddof = 0)) 
##  
##print ("\nsignaltonoise ratio for arr1 : ",  
##       signaltonoise(arr1, axis = 1, ddof = 0)) 
#  
#print ("\nsignaltonoise ratio for arr1 : ",  
#       signaltonoise(arr2, axis = 0, ddof = 0))  

