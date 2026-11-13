import pickle
import numpy as np

def modelo(x_in, N, M):

    # Función para calcular las predicciones del modelo MPM
    def MPM_out(x, a_opt, N, M):
        """
        Calcula las salidas predichas utilizando los coeficientes óptimos.
        
        Args:
            x: Entrada de la señal en formato complejo.
            a_opt: Coeficientes complejos a_n calculados.
            N: Grado del polinomio.
            M: Número de términos de memoria.
            
        Returns:
            y_pred: Salida predicha en formato complejo.
        """
        x = np.array(x, dtype=np.complex128)
        
        # Inicializa la matriz de diseño
        L = len(x)
        X = np.zeros((L, (N+1) * (M+1)), dtype=np.complex128)
        
        # Construir la matriz de características
        for n in range(L):
            for m in range(M+1):
                if n - m >= 0:
                    for k in range(N+1):
                        X[n, m * N + k] = x[n - m] * (np.abs(x[n - m]) ** k)
        
        # Calcular las salidas predichas
        y_pred = X @ a_opt
        
        return y_pred

    # Función para generar predicciones y validar el modelo MPM
    def MPM_model(x, a, N, M):
        y_pred = MPM_out(x, a, N, M)
        return y_pred


    # Función para calcular las salidas de la etapa NL 
    def etapa_NL(x, b):
        """
        Calcula la salida de la etapa no lineal del modelo Hammerstein.
        
        :param x: Lista o array de datos de entrada en formato complejo.
        :param b: Coeficientes del polinomio (vector b) en formato complejo.
        :return: Salida de la etapa no lineal en formato complejo.
        """
        # Evaluar el polinomio dado por coeficientes b en los puntos x
        ynl = np.polyval(b[::-1], x)

        return ynl

    # Función para calcular las salidas de la etapa LTI del modelo
    def etapa_LTI(ynl, h):
        """
        Calcula las salidas predichas de la etapa lineal del modelo Hammerstein.
        
        :param ynl: Lista o array de salidas de la etapa no lineal en formato complejo.
        :param h: Coeficientes del filtro lineal en formato complejo.
        :return: Salidas predichas en formato complejo.
        """

        # Calcular el número de observaciones
        L = len(ynl)
        
        # Calcular la profundidad de memoria
        M = len(h) - 1
        
        # Construir la matriz de diseño YNL con valores complejos
        YNL = np.zeros((L, M + 1), dtype=complex)
        
        for i in range(M + 1):
            YNL[i:, i] = ynl[:L - i]
        
        # Calcular las salidas predichas usando la multiplicación de matrices
        ylti = YNL @ h
        return ylti

    # Función para generar predicciones y validar el modelo Hammerstein
    def hammerstein_model(x, b, h):
        """
        Calcula el error de las predicciones del modelo Hammerstein utilizando los coeficientes b y h.

        :param x: Datos de entrada.
        :param y: Datos reales de salida.
        :param b: Coeficientes de la etapa no lineal.
        :param h: Coeficientes de la etapa lineal.
        :param error_metric: Métrica de error a utilizar ('mse', 'nmse', 'nmse_dB', etc.).
        :return: Error calculado entre las salidas predichas y las reales.
        """
        # Obtener la salida de la etapa no lineal
        ynl = etapa_NL(x, b)
        
        # Obtener la salida de la etapa lineal
        ylti = etapa_LTI(ynl, h)
        
        return ylti


    # Función para generar predicciones y validar el modelo Wiener
    def wiener_model(x, h, b):
        """
        Calcula el error de las predicciones del modelo Wiener utilizando los coeficientes h y b.

        :param x: Datos de entrada.
        :param y: Datos reales de salida.
        :param h: Coeficientes de la etapa lineal.
        :param b: Coeficientes de la etapa no lineal.
        :param error_metric: Métrica de error a utilizar ('mse', 'nmse', 'nmse_dB', etc.).
        :return: Salida predicha y error calculado entre las salidas predichas y las reales.
        """
        # Obtener la salida de la etapa lineal
        ylti = etapa_LTI(x, h)
        
        # Obtener la salida de la etapa no lineal
        ynl = etapa_NL(ylti, b)

        return ynl


    N = 6
    M = 7


    with open('coeficientes.pkl', 'rb') as file:
                datos = pickle.load(file)

    # Acceder a los valores de a, b y h
    a = datos['a']
    b = datos['b']
    h = datos['h']

    modelo = 'Hammerstein'

    # Calcular las predicciones con los coeficientes
    if modelo == 'MPM':
        y_pred = MPM_model(x_in, a, N, M)
    elif modelo == 'Hammerstein':
        y_pred = hammerstein_model(x_in, b, h) 
    elif modelo == 'Wiener':
        y_pred = wiener_model(x_in, h, b)
    
    return y_pred