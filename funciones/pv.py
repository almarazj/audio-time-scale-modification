import numpy as np
from scipy import signal

def pv(x, fs, alpha, n_ventana, Hs_arm):
    
    output_length = int(len(x)/alpha)
    y = np.zeros(output_length)
    # Se inicializa el vector de la señal escalada.

    Hs = Hs_arm   # Número de muestras del cuadro de síntesis.
    Ha = alpha*Hs # Número de muestras del cuadro de síntesis.
    k, _, Zxx = signal.stft(x,nperseg=n_ventana, noverlap=n_ventana-Ha, padded=True)
    # Se computa la STFT de la señal de entrada.
    
    pha = np.angle(Zxx) # Fase de la STFT
    omega = 2*np.pi*k

    pha_pred = np.zeros(pha.shape) # Fase predecida.
    pha_err = np.zeros(pha.shape)  # Error entre la fase y la fase predecida.
    pha_mod = np.zeros(pha.shape)  # Fase modificada
    IF = np.zeros(pha.shape)       # Frecuencia instantánea.
    Y = np.zeros(Zxx.shape, dtype = complex) #STFT de salida.
    
    Y[:,0] = Zxx[:,0]
    pha_pred[:,0] = pha[:,0]
    pha_mod[:,0] = pha[:,0]
    IF[:,0] = omega
    # Se inicializa la primer columna de cada matriz

    for i in range(pha.shape[1]-1):
        # Propagación de fase
        # Driedger, J., & Müller, M. (2016). 
        # A Review of Time-Scale Modification of Music Signals.

        pha_pred[:,i+1] = pha[:,i] + omega*Ha

        pha_err[:,i+1] = pha[:,i+1] - pha_pred[:,i+1]
        pha_err[:,i+1] = (pha_err[:,i+1] + np.pi) % (2 * np.pi) - np.pi

        IF[:,i+1] = (omega+pha_err[:,i+1]/Ha)

        pha_mod[:,i+1] = pha_mod[:,i] + IF[:,i]*Hs
            
        theta = pha_mod[:,i+1] - pha[:,i+1]

        phasor = np.exp(1j * theta)

        Y[:,i+1] = phasor * Zxx[:,i+1]

    _, y_arm = signal.istft(Y, fs = fs, window = 'hann',
                     nperseg = n_ventana, noverlap = n_ventana - Hs)
    # Se computa la ISTFT del espectrograma de fase modificada.

    if len(y_arm)>len(y):          # Caso de que la ISTFT devuelva
        y = y_arm[0:len(y)]        # un vector de mayor dimensión
    else:                          # a la esperada.             
        y[0:int(len(y_arm))]=y_arm # Caso de que la ISTFT devuelva
                                   # un vector de menor dimensión a 
                                   # la esperada.

    t = np.linspace(0,len(y)/fs, len(y))

    return t, y
