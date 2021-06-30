#%%
from numpy.lib.function_base import average, percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.cluster import KMeans


#%%
'''
-------------------------
CARGA DE DATASET BLINKING
-------------------------
'''
signals = pd.read_csv(  'data/blinking.dat', delimiter=' ', 
                        names = ['timestamp','counter','eeg','attention','meditation','blinking']
                        )

print('Information:')
print(signals.head())

data=signals.values

#%%
#Me quedo con la columna corrsepondiente a eeg
eeg = data[:,2]
# %%
print("Largo 1 %2d" % len(eeg))

# %%
plt.plot(eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Original EEG Signal')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eeg)])
plt.savefig('signal.png')
plt.show()



# %%
# La operación de convolución permite implementar el suavizado del Moving Average
windowlength = 10
avgeeg = np.convolve(eeg, np.ones((windowlength,))/windowlength, mode='same')
# %%
# El kernel/máscara está compuesto de 10 valores de 1/10.  Cuando esos valores se suman para cada posición, implica que se reemplaza el valor por el promedio
# de los 5 valores anteriores y 4 posteriores.

x = avgeeg
kmeans = KMeans(n_clusters=3).fit(x.reshape(-1,1))
kmeans.predict(x.reshape(-1,1))

# %%
# %%
yValues = x
xValues = np.arange(0,len(x),1)
colorValues = kmeans.labels_
# values of x
x = xValues 
# values of y
y = yValues
  
# empty list, will hold color value
# corresponding to x
col =[]
  
for i in range(0, len(x)):
    if colorValues[i]==0:
        col.append('blue') 
    elif colorValues[i]==1:
        col.append('green') 
    else:
        col.append('magenta') 
  
for i in range(len(x)):
      
    # plotting the corresponding x with y 
    # and respective color
    plt.scatter(x[i], y[i], c = col[i], s = 10,
                linewidth = 0)
      
  
plt.show()
# %%
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
print("Blinking counter: {}".format(contador))
filtro_eeg=np.asarray(filtro_eeg)
plt.figure(figsize=(16,5))
plt.plot(filtro_eeg,color="blue")
plt.title("Blinking Filter",size=20)
plt.ylabel("Class",size=10)
plt.xlabel("Timepoint",size=10)
plt.show()

# %%
