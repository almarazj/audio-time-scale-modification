import numpy as np
import soundfile as sf
from funciones.hpss import *
from funciones.ola import *
from funciones.pv import *

#--------Obtención del archivo de audio en formato wav--------
x, fs = sf.read('audio files\\DrumSolo.wav')

#-------------Definición de parámetros de entrada-------------

alpha = 0.75 # factor de escala.

n_win_hpss = 2048 # Tamaño de la ventana para el HPSS.

n_win_arm = 2048 # Tamaño de la ventana para el Phase Vocoder.
Hs_arm = 512 # Numero de muestras del cuadro de síntesis. 

n_win_perc = 512 # Tamaño de la ventana para el OLA.

#----------Separación de la señal armónica-percusiva----------
_, _, x_arm, x_perc = hpss(x,fs,25,50, n_win_hpss)

#-------------Escalado de la componente percusiva-------------
t_perc, y_perc = ola(x_perc, 1/alpha, n_win_perc)

#-------------Escalado de la componente armónica--------------
t_arm, y_arm = pv(x_arm, fs, 1/alpha, n_win_arm, Hs_arm)

#------------------Suma de las componentes--------------------
y = y_arm+y_perc


x_RMS = np.sqrt(np.mean(x**2))  # Valor RMS de la señal de entrada.
y_RMS = np.sqrt(np.mean(y**2))  # Valor RMS de la señal de salida.

y = y*(x_RMS/y_RMS)             # Se iguala el nivel de la señal de
                                # salida con el de la señal de entrada.

sf.write('audio files\\DrumSolo\\DrumSolo_0.75_.wav', y, fs)
# Se guarda la señal escalada



