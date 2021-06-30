"""
The goal of this program is to detect blinks.
The user must input the eeg signals with two require arguments:
    1 - file_path
    2 - channel: number of column where it is the corresponding value measure by the sensor

The user has also the chance to pass some optional arguments: 
    -w int: size of the window to calculate the moving average.
    -n int: number to increase (n >1) or decrease (0<n<1) the upper and lower threshold used to detec peaks
Three ways to calculate the thresholds
    -o1: thresholds = mean +- n standart deviations
    -02: thresholds = median +- n standart deviations
    -o3: thresholds = median +- n interquartil distance
    If this argument is not passed, default is option 1.
Last optional argument use a clustering technique to filter the blinks
    -clustering

"""
# %%
import sys
from numpy.lib.function_base import average, percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import argparse
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='This is a program to detect blinkings. You can pass some options.')
parser.add_argument("inputdata", help="path to input data file")
parser.add_argument("col", help="column of data to find peaks")
parser.add_argument("-w", "--window", help="select mean moving window, 10 is the default value", default="10")
parser.add_argument("-o1", "--option1", help="Threshold = int(mean()+n*std()) ", action="store_true")
parser.add_argument("-o2", "--option2", help="Threshold = int(median()+n*std()) ", action="store_true")
parser.add_argument("-o3", "--option3", help="Threshold = int(median()+n*std()) ", action="store_true")
parser.add_argument("-n", "--multiplier", help="select n multiplier in threshold", default="3")
parser.add_argument("-clustering", "--clustering", help="This option calculates blinks doing clustering ", action="store_true")
args = parser.parse_args()

ncol=int(args.col)
w=int(args.window)
n=int(args.multiplier)



#%%
'''
-------------------------
LOAD OF DATASET
-------------------------
'''
signals = pd.read_csv(  str(args.inputdata), delimiter=' ', 
                        names = ['timestamp','counter','eeg','attention','meditation','blinking']
                        )

print('Information:')
print(signals.head())
data=signals.values
eeg = data[:,ncol]
print("Number of samples measured %2d" % len(eeg))

'''
-------------------------
PLOT OF ORIGINAL SIGNAL
-------------------------
'''
plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Original EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.savefig('signal.png')
plt.show()

# La operación de convolución permite implementar el suavizado del Moving Average
windowlength = w
avgeeg = np.convolve(eeg, np.ones((windowlength,))/windowlength, mode='same')
'''
-------------------------
PLOT OF SMOOTHED SIGNAL
-------------------------
'''
plt.plot(avgeeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Smoothed EEG Signal')     
plt.ylim([-2000, 2000]);
plt.xlim([0,len(avgeeg)])
plt.savefig('smoothed.png')
plt.show()
'''
-------------------------
PLOT OF PEAKS
-------------------------
'''
peaks, _ = find_peaks(avgeeg, height=200,distance=128)
plt.plot(avgeeg)
plt.plot(peaks, avgeeg[peaks], "x")
plt.plot(np.zeros_like(avgeeg), "--", color="gray")
plt.show()

'''
------------------------------
4 WAYS OF FILTERING THE SIGNAL
------------------------------
'''
print("----TIME POSITION OF BLINKS----")
if args.clustering:
    kmeans = KMeans(n_clusters=3).fit(avgeeg.reshape(-1,1))
    colorValues = kmeans.labels_
    x = np.arange(0,len(avgeeg),1) 
    y = avgeeg
    filtro_eeg=[]
    contador=0
    for i in range(len(avgeeg)):
        if colorValues[i]==0:
            filtro_eeg.append(0)
        elif colorValues[i]==1:
            filtro_eeg.append(1)
            if colorValues[i-1] != colorValues[i]:
                print(i)
                contador=contador+1
        elif colorValues[i]==2:
            filtro_eeg.append(-1)
            if colorValues[i-1] != colorValues[i]:
                print(i)
                contador=contador+1
        else:
            filtro_eeg.append(0)
    print("----BLINKS COUNT----")
    print("Blinking counter: {}".format(contador))
    filtro_eeg=np.asarray(filtro_eeg)
    plt.figure(figsize=(16,5))
    plt.plot(filtro_eeg,color="blue")
    plt.title("Blinking Filter",size=20)
    plt.ylabel("Class",size=10)
    plt.xlabel("Timepoint",size=10)
    plt.show()
else:
    if args.option1:
        delta=n*avgeeg.std()
        upper_threshold=int(avgeeg.mean()+delta)
        lower_threshold=int(avgeeg.mean()-delta)
    elif args.option2:
        delta=n*avgeeg.std()
        upper_threshold=int(np.median(avgeeg)+delta)
        lower_threshold=int(np.median(avgeeg)-delta)
    elif args.option3:
        delta=n*(np.percentile(avgeeg,75)-np.percentile(avgeeg,25))
        upper_threshold= int(np.median(avgeeg)+delta)
        lower_threshold= int(np.median(avgeeg)-delta)
    else:
        delta=n*avgeeg.std()
        upper_threshold=int(avgeeg.mean()+delta)
        lower_threshold=int(avgeeg.mean()-delta)
    print("Upper Threshold: {}".format(upper_threshold))
    print("Lower Threshold: {}".format(lower_threshold))
    plt.figure(figsize=(12,5))
    plt.plot(avgeeg,color="green")
    plt.plot(np.full(len(avgeeg),upper_threshold),'r--')
    plt.plot(np.full(len(avgeeg),lower_threshold),'r--')
    plt.ylabel("Amplitude",size=10)
    plt.xlabel("Timepoint",size=10)
    plt.title("avgeeg Series with control limits",size=20)
    plt.annotate("Upper Threshold",xy=(500,upper_threshold+10),color="red")
    plt.annotate("Lower Threshold",xy=(500,lower_threshold+10),color="red")
    plt.show()
    '''
    Now the EEG data is filtered to produce a new output, assigning 1, greater than the upper limit, 0 between lower and upper
    limit, and -1, under the lower limit.  In order to determine the number of valid events, changes from 0-1 will be counted
    as a possible blinking event.
    '''
    filtro_eeg=[]
    contador=0
    for i in range(len(avgeeg)):
        if i==0:
            filtro_eeg.append(0)
        elif avgeeg[i]>upper_threshold:
            filtro_eeg.append(1)
            if avgeeg[i-1]<=upper_threshold:
                print(i)
                contador=contador+1
        elif avgeeg[i]<lower_threshold:
            filtro_eeg.append(-1)
        else:
            filtro_eeg.append(0)
    print("Blinking counter: {}".format(contador))
    filtro_eeg=np.asarray(filtro_eeg)
    plt.figure(figsize=(16,5))
    plt.plot(filtro_eeg,color="blue")
    plt.title("Blinking Filter",size=20)
    plt.ylabel("Class",size=10)
    plt.xlabel("Timepoint",size=10)
    plt.show()

