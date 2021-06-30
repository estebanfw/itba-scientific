"""
==================
Final Assignment
==================


"""
# %%
print(__doc__)
from numpy.lib.function_base import average, percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('ECD')
print('Este ejercicio tiene dos maneras de resolverse.')
print('Lo tiene que tener listo para el fin de la Cuarentena')

print('Opción B: elijan una (al menos) pregunta e intentan implementar una solución, codificando en R, Java o python.')

print('0 - Construyan una alternativa para detectar pestañeos (blinking.dat) y trabajen sobre el dataset de pestañeos para simular y testear el abordaje propuesto.')
print('1 - De las señales del EPOC Emotiv que obtuvimos de SUJETO, intenten estudiar las señales detectando: los pestañeos sobre F8 y F7, el momento donde el sujeto cierra los ojos, donde abre y cierra la boca, donde mueve la cabeza haciendo Roll, y donde mueve la cabeza haciendo YAW.')
print('2 - Sobre los datos de MNIST, intenten luego de clusterizar armar un clasificador.')
print('3 - Busquen un dataset de internet público de señales de sensores.  ¿Cómo lo abordarían exploratoriamente, qué procesamiento y qué análisis harían?')
print('4 - Prueben alternativas para mejorar la clasificación de las ondas alfa.')
print('5 - ¿Que feature utilizarian para mejorar la clasificacion que ofrece Keras con MLP para las series de tiempo?')
print('6 - Suban un snippet que aborde alguna problemática de solución, implementé algún otro método de clasificación o de análisis, sobre los datos registrados en este repositorio.')


# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO



#%%
'''
====================================
EJERCICIO 0 - DETECCIÓN DE PESTAÑEOS
====================================
'''
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

# Filtro de todos los valores solo aquellos que son efectivamente mayores a 50
eegf1 = eeg[eeg>50]

# Muchas veces lo que me interesa es saber los índices (que en series de tiempo representan el tiempo) donde el filtro es positivo
# Esto se hace con el comando np.where
idxeeg1f = np.where( eeg > 50 )

# Filtro los valores que son mayores a 10 y menores que -40
eegf2 = eeg[np.logical_or(eeg>10,eeg<-40)] 

print("Largo 1 %2d" % len(eeg))
print("Largo 2 %2d" % len(eegf1))
print("Largo 3 %2d" % len(eegf2))
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

plt.plot(avgeeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Smoothed EEG Signal')     
plt.ylim([-2000, 2000]);
plt.xlim([0,len(avgeeg)])
plt.savefig('smoothed.png')
plt.show()
# %%
from scipy.signal import find_peaks
# %%
peaks, _ = find_peaks(avgeeg, height=200,distance=128)
plt.plot(avgeeg)
plt.plot(peaks, avgeeg[peaks], "x")
plt.plot(np.zeros_like(avgeeg), "--", color="gray")
plt.show()

# %%
#Find the threshold values to determine what is a blinking and what is not
delta=3*avgeeg.std()
#delta=2*(np.percentile(avgeeg,75)-np.percentile(avgeeg,25))-avgeeg.mean()+np.median(avgeeg)
#delta=np.max(avgeeg)*.50-avgeeg.mean()
umbral_superior=int(avgeeg.mean()+delta)
print("Upper Threshold: {}".format(umbral_superior))
umbral_inferior=int(avgeeg.mean()-delta)
print("Lower Threshold: {}".format(umbral_inferior))
plt.figure(figsize=(12,5))
plt.plot(avgeeg,color="green")
plt.plot(np.full(len(avgeeg),umbral_superior),'r--')
plt.plot(np.full(len(avgeeg),umbral_inferior),'r--')
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("avgeeg Series with control limits",size=20)
plt.annotate("Upper Threshold",xy=(500,umbral_superior+10),color="red")
plt.annotate("Lower Threshold",xy=(500,umbral_inferior+10),color="red")
plt.show()
# %%
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
    elif avgeeg[i]>umbral_superior:
        filtro_eeg.append(1)
        if avgeeg[i-1]<=umbral_superior:
            print(i)
            contador=contador+1
    elif avgeeg[i]<umbral_inferior:
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

