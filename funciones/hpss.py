from scipy import signal
from scipy.ndimage import median_filter
import numpy as np

def hpss(x, fs, filtro_hor, filtro_ver, win_size):

    # Se computa la STFT de la señal
    _, _, X = signal.stft(x, fs, nperseg = win_size, padded = True, noverlap=win_size//2)

    mag_X = np.abs(X) # Magnitud de la STFT

    mag_hor = median_filter(np.real(mag_X), size=[1, filtro_hor], mode='reflect')
    mag_ver = median_filter(np.real(mag_X), size=[filtro_ver,1], mode='reflect')
    # Se obtiene el espectro aumentado horizontal y verticalmente

    mask_hor = mag_hor > mag_ver  # Máscaras horizontales
    mask_ver = mag_hor <= mag_ver # y verticales

    X_hor = mask_hor * X # Se aplica cada máscara
    X_ver = mask_ver * X # a la STFT obtenida

    x_arm = np.zeros(x.shape)
    x_perc = np.zeros(x.shape)
    # Se inicializan los vectores de la señal de salida

    t_arm, x_arm = signal.istft(X_hor, nperseg = win_size, noverlap=win_size//2)
    t_perc, x_perc = signal.istft(X_ver, nperseg = win_size, noverlap=win_size//2)
    # Se computa la ISTFT, obteniendo las componentes de la señal por separado.
    
    return t_arm, t_perc, x_arm, x_perc
