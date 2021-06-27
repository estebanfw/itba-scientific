"""
==================
Final Assignment
==================


"""
# %%
print(__doc__)
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


'''
El experimento para el Ejercicio (1) está en el directorio data/experimentosujeto.dat

Sujeto #1 se colocó el dispositivo de captura de señales de EEG EPOC Emotiv.  Cuatro canales se habilitaron F7, F8 frontales y O1,O2 occipitales.
El dispositvo además tiene información de dos IMUs, en Gyro_x y Gyro_y.
La persona estuvo sentada durante 5 minutos aproximadamente.  Durante diferentes períodos de tiempo realizó las siguientes acciones

* Movimiento de la cabeza hacia los laterales (Yaw)
* Movimiento de la cabeza hacia adelante y atrás (pitch)
* Movimiento de la cabeza hacia los lados (llevando las orejas a los hombros) (roll)
* Pestañeo voluntario intermitente
* Apertura y cierre de la boca.
* Cerró los ojos.
* Permaneció inmovil mirando un punto fijo (y pestañando naturalmente).

El formato de los datos es

        "COUNTER",
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        "GYRO_X",
        "GYRO_Y",
        "RESERVED",
        "RESERVED",
        "RESERVED",
        "RESERVED",
        "RESERVED"

Los datos buenos que tomamos deberían ser F7 y F8, GYRO_X y GYRO_Y.

'''



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
# Operacion de Convolucion matemática
print('Original Signal')
print([1,2,3])
print('Kernel:')
print([-1,1,-1])
convolvedsignal = np.convolve([1,2,3],[-1,1,-1], 'same')
print('Output Signal')
print(convolvedsignal)
# %%
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
plt.plot(eegf1,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Original EEG Signal - Filter 1')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eegf1)])
plt.savefig('signal.png')
plt.show()

# %%
plt.plot(eegf2,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Original EEG Signal - Filter 2')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(eegf2)])
plt.savefig('signal.png')
plt.show()

# %%


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

windowlength = 5
avgeeg = np.convolve(eeg, np.ones((windowlength,))/windowlength, mode='same')
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
