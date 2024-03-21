import numpy as np

def ola(x, alpha, n_ventana):

    # Parámetros de entrada.
    N = int(n_ventana)
    Hs = n_ventana//2
    Ha = alpha*Hs
    w = np.hanning(N)

    output_length = int(len(x)/alpha)
    y = np.zeros(output_length)
    # Se inicializa el vector de la señal escalada.

    for i in range(0, int(len(x) - N), int(Ha)):
        index = int(i/alpha)
        if index + N > output_length: # Se mapea el último cuadro.
            y[index:output_length] += x[i:i+output_length-index]*w[:output_length-index]
            break
        y[index:index+N] += x[i:i+N]*w # Se mapea cada cuadro de análisis
                                       # a la señal de salida escalada.

    t = np.linspace(0,len(y)/44100, len(y))

    return t, y 