import numpy as np
# from funciones import *
from tqdm import tqdm
import pickle
import scipy.io

import sys
import os
# Agregar el directorio padre a sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Importar funciones desde el script en volterra
from metodos_tradicionales.funciones import *

# Función para calcular los coeficientes complejos del modelo MPM
def MPM_a(x, y, N, M):
    """
    Implementa el modelo polinomial con memoria.
    
    Args:
        x: Entrada de la señal en formato complejo.
        y: Salida de la señal en formato complejo.
        N: Grado del polinomio.
        M: Número de términos de memoria.
        
    Returns:
        a_opt: Coeficientes complejos a_n más óptimos.
    """
    x = np.array(x, dtype=np.complex128)
    y = np.array(y, dtype=np.complex128)
    
    # Inicializa la matriz de características
    L = len(x)
    X = np.zeros((L, (N+1) * (M+1)), dtype=np.complex128)
    
    # Construir la matriz de diseño
    for n in range(L):
        for m in range(M+1):
            if n - m >= 0:
                for k in range(N+1):
                    X[n, m * N + k] = x[n - m] * (np.abs(x[n - m]) ** k)
    
    # Resolver el sistema de ecuaciones para encontrar a_n óptimos
    # y = X * a
    a_opt, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    return a_opt

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

# Funciones para entrenar el modelo MPM
def MPM_train(x, y, N, M, error_metric):
    a = MPM_a(x, y, N, M)
    y_pred = MPM_out(x, a, N, M)
    error = calculate_error(y, y_pred, error_metric)
    return a, error

# Función para generar predicciones y validar el modelo MPM
def MPM_model(x, y, a, N, M, error_metric):
    y_pred = MPM_out(x, a, N, M)
    error = calculate_error(y, y_pred, error_metric)
    return y_pred, error




# Función para calcular los coeficientes de la etapa NL
def coef_etapa_NL(x, y, N):
    """
    Calcula los coeficientes de la etapa no lineal usando mínimos cuadrados.
    
    :param x: Lista o array de datos de entrada en formato complejo.
    :param y: Lista o array de datos de salida en formato complejo.
    :param N: Grado de no linealidad.
    :return: Coeficientes b del modelo no lineal.
    """
    L = len(x)  # Número de observaciones
    # Construir la matriz de diseño X con valores complejos
    X = np.zeros((L, N + 1), dtype=complex)
    for i in range(N + 1):
        X[:, i] = x ** i
    
    # Calcular los coeficientes b usando mínimos cuadrados
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    return b


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

# Función para calcular los coeficientes de la etapa LTI
def coef_etapa_LTI(ynl, y, M):
    """
    Calcula los coeficientes de la etapa lineal del modelo usando mínimos cuadrados.
    
    :param ynl: Lista o array de salidas de la etapa no lineal en formato complejo.
    :param y: Lista o array de datos de salida en formato complejo.
    :param M: Profundidad de memoria (orden del filtro).
    :return: Coeficientes h del filtro lineal en formato complejo.
    """
    # Calcular número de observaciones
    L = len(ynl)
    
    # Construir la matriz de diseño YNL con valores complejos
    YNL = np.zeros((L, M + 1), dtype=complex)
    
    for i in range(M + 1):
        YNL[i:, i] = ynl[:L - i]
    
    # Calcular los coeficientes h usando mínimos cuadrados
    h, _, _, _ = np.linalg.lstsq(YNL, y, rcond=None)
    
    return h

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

# Función para entrenar el modelo Hammerstein
def hammerstein_model_train(x, y, N, M, error_metric):
    """
    Entrena el modelo Hammerstein y calcula el error basado en la métrica especificada.
    
    :param x: Datos de entrada.
    :param y: Datos de salida.
    :param N: Grado del polinomio de la etapa no lineal.
    :param M: Profundidad de memoria del filtro lineal.
    :param error_metric: Métrica de error a utilizar ('mse', 'nmse', 'nmse_db', etc).
    :return: Coeficientes b, coeficientes h, y error calculado según la métrica.
    """
    # Calcular coeficientes b de la etapa no lineal
    b = coef_etapa_NL(x, y, N)
    # Obtener la salida de la etapa no lineal
    ynl = etapa_NL(x, b)
    # Calcular coeficientes h de la etapa lineal
    h = coef_etapa_LTI(ynl, y, M)
    # Obtener la salida predicha de la etapa lineal
    ylti = etapa_LTI(ynl, h)
    # Calcular el error usando la métrica especificada
    error = calculate_error(y, ylti, error_metric)
    
    return b, h, error

# Función para generar predicciones y validar el modelo Hammerstein
def hammerstein_model(x, y, b, h, error_metric):
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
    
    # Calcular el error usando la métrica especificada
    error = calculate_error(y, ylti, error_metric)
    return ylti, error


# Función para entrenar el modelo Wiener
def wiener_model_train(x, y, N, M, error_metric):
    """
    Entrena el modelo Wiener y calcula el error basado en la métrica especificada.
    
    :param x: Datos de entrada.
    :param y: Datos de salida.
    :param N: Grado del polinomio de la etapa no lineal.
    :param M: Profundidad de memoria del filtro lineal.
    :param error_metric: Métrica de error a utilizar ('mse', 'nmse', 'nmse_db', etc).
    :return: Coeficientes h, coeficientes b, y error calculado según la métrica.
    """
    # Calcular coeficientes h de la etapa lineal
    h = coef_etapa_LTI(x, y, M)
    # Obtener la salida de la etapa lineal
    ylti = etapa_LTI(x, h)
    # Calcular coeficientes b de la etapa no lineal
    b = coef_etapa_NL(ylti, y, N)
    # Obtener la salida predicha de la etapa no lineal
    ynl = etapa_NL(ylti, b)
    # Calcular el error usando la métrica especificada
    error = calculate_error(y, ynl, error_metric)
    
    return h, b, error

# Función para generar predicciones y validar el modelo Wiener
def wiener_model(x, y, h, b, error_metric):
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
    
    # Calcular el error usando la métrica especificada
    error = calculate_error(y, ynl, error_metric)
    return ynl, error


def entrenar_validar_modelo(x_train, y_train, x_valid, y_valid, valores_N, valores_M, error_metric, modelo):
    """
    Entrena y valida el modelo utilizando valores de N y M, y calcula los errores de entrenamiento y validación.

    Args:
        x_train (ndarray): Señales de entrada para el entrenamiento.
        y_train (ndarray): Señales de salida para el entrenamiento.
        x_valid (ndarray): Señales de entrada para la validación.
        y_valid (ndarray): Señales de salida para la validación.
        valores_N (list): Lista de valores para N.
        valores_M (list): Lista de valores para M.
        error_metric (function): Función que calcula la métrica de error.
        valores_a (list): Lista donde se almacenarán los coeficientes.

    Returns:
        tuple: 
            - error_train (ndarray): Matriz de errores del entrenamiento.
            - error_valid (ndarray): Matriz de errores de validación.
            - valores_a, valores_b, valores_h (list): Lista actualizada con los coeficientes.
    """
    # Inicializar matrices para almacenar errores
    error_train = np.zeros((len(valores_N), len(valores_M)))
    error_valid = np.zeros((len(valores_N), len(valores_M)))

    # Crear una barra de progreso para el entrenamiento
    total_iterations = len(valores_N) * len(valores_M)

    # Listas para almacenar los coeficientes a
    valores_a = []
    valores_b = []
    valores_h = []

    with tqdm(total=total_iterations, desc=f'Entrenamiento y Validación del Modelo {modelo}') as pbar:
        # Iterar sobre valores de N y M
        for i, N in enumerate(valores_N):
            for j, M in enumerate(valores_M):
                try:
                    if modelo == 'MPM':
                        # Entrenamiento del modelo
                        a, error_train[i, j] = MPM_train(x_train, y_train, N, M, error_metric)
                        # Validación del modelo
                        _, error_valid[i, j] = MPM_model(x_valid, y_valid, a, N, M, error_metric)
                        # Almacenar los coeficientes b y h
                        valores_a.append(a)

                    if modelo == 'Hammerstein':
                        # Entrenamiento del modelo
                        b, h, error_train[i, j] = hammerstein_model_train(x_train, y_train, N, M, error_metric)
                        # Validación del modelo
                        _, error_valid[i, j] = hammerstein_model(x_valid, y_valid, b, h, error_metric)
                        # Almacenar los coeficientes b y h
                        valores_b.append((b))
                        valores_h.append((h))

                    if modelo == 'Wiener':
                        # Entrenamiento del modelo
                        h, b, error_train[i, j] = wiener_model_train(x_train, y_train, N, M, error_metric)
                        # Validación del modelo
                        _, error_valid[i, j] = wiener_model(x_valid, y_valid, h, b, error_metric)
                        # Almacenar los coeficientes b y h
                        valores_b.append((b))
                        valores_h.append((h))

                except Exception as e:
                    print(f"Error en N={N}, M={M}: {e}")
                    error_train[i, j] = np.nan  # Asignar NaN en caso de error
                    error_valid[i, j] = np.nan  # Asignar NaN en caso de error

                # Actualizar la barra de progreso
                pbar.update(1)

    crear_graficos_3d(error_train, error_valid, valores_N, valores_M, error_metric)

    return valores_a, valores_b, valores_h

def test_modelo(valores_N, valores_M, N, M, valores_a, valores_b, valores_h, x_test, y_test, error_metric, modelo):
    # Obtención de coeficientes óptimos para N y M elegidos
    if modelo == 'MPM':
        a = obtener_coeficientes_optimos(valores_N, valores_M, N, M, valores_a)
        b=0
        h=0
        # Etapa de prueba con los coeficientes a
        _, error_test = MPM_model(x_test, y_test, a, N, M, error_metric)
        print("-------------------------------")
        print("valores coeficientes complejos (a):")
        print(a)
        print("-------------------------------")
        print(f"{error_metric} test para N={N} y M={M} del Modelo {modelo} = {error_test}")
    elif modelo == 'Hammerstein' or 'Wiener':
        a=0
        b = obtener_coeficientes_optimos(valores_N, valores_M, N, M, valores_b)
        h = obtener_coeficientes_optimos(valores_N, valores_M, N, M, valores_h)

        if modelo == 'Hammerstein':
            _, error_test = hammerstein_model(x_test, y_test, b, h, error_metric)
        if modelo == 'Wiener':
            _, error_test = wiener_model(x_test, y_test, h, b, error_metric)

        print("-------------------------------")
        print("valores b:")
        print(b)
        print("-------------------------------")
        print("valores h:")
        print(h)
        print("-------------------------------")
        print(f"{error_metric} test para N={N} y M={M} del Modelo {modelo} = {error_test}")

    return a, b, h



def etapas_modelo(entrenar,x_train, y_train, x_valid, y_valid, valores_N, valores_M, N, M, x_test, y_test, error_metric, modelo):
    if entrenar == 'si':
        # Entrenamiento y validación
        valores_a, valores_b, valores_h = entrenar_validar_modelo(x_train, y_train, x_valid, y_valid, valores_N, valores_M, error_metric, modelo)
        # Prueba y obtención de coeficientes óptimos para N y M elegidos 
        a, b, h = test_modelo(valores_N, valores_M, N, M, valores_a, valores_b, valores_h, x_test, y_test, error_metric, modelo)
        
        # Guardar los datos en un archivo
        with open('coeficientes.pkl', 'wb') as file:
            pickle.dump({'a': a, 'b': b, 'h': h}, file)

        print("Datos guardados exitosamente en 'coeficientes.pkl'")
        # generar ecuaciones
        # equations_to_file(a, N, M, f'{modelo}/eq_{modelo}.txt')
        
    else:
        # Cargar los datos desde el archivo
        with open('coeficientes.pkl', 'rb') as file:
            datos = pickle.load(file)

        # Acceder a los valores de a, b y h
        a = datos['a']
        b = datos['b']
        h = datos['h']

    # Cargar los datos desde el archivo pickle
    with open('coeficientes.pkl', 'rb') as file:
        datos = pickle.load(file)

    # Guardar los datos en un archivo .mat
    scipy.io.savemat('coeficientes.mat', datos)
    
    return a, b, h


def modelos(x_original, y_original, a, b, h, N, M, error_metric, modelo):
    # Calcular las predicciones con los coeficientes
    if modelo == 'MPM':
        y_pred, error_pred = MPM_model(x_original, y_original, a, N, M, error_metric)
    elif modelo == 'Hammerstein':
        y_pred, error_pred = hammerstein_model(x_original, y_original, b, h, error_metric) 
    elif modelo == 'Wiener':
        y_pred, error_pred = wiener_model(x_original, y_original, h, b, error_metric)
    
    return y_pred, error_pred

def usar_modelo(tipo, entrenar, valores_N, valores_M, N, M, error_metric, modelo):
    # Leer datos
    x_train, y_train, x_valid, y_valid, x_test, y_test, time, x_original, y_original = leer_datos(tipo)

    # Entrenamiento, validación y prueba
    a, b, h = etapas_modelo(entrenar,x_train, y_train, x_valid, y_valid, valores_N, valores_M, N, M, x_test, y_test, error_metric, modelo)
    
    # def equations_to_file(a, N, M, filename):
    #     """
    #     Calcula las ecuaciones de salida y las guarda en un archivo.
        
    #     Args:
    #         a: Coeficientes complejos a_n.
    #         N: Grado del polinomio.
    #         M: Número de términos de memoria.
    #         filename: Nombre del archivo donde se guardarán las ecuaciones.
    #     """
    #     with open(filename, 'w') as file:

    #         # Inicializar la variable para almacenar la parte real
    #         y_REAL = ""

    #         # Generar la parte real
    #         for k in range(N+1):
    #             for m in range(M+1):
    #                 y_REAL += f' + ({ (a[m * N + k].real):.8f})*(_vin[n-{m}])**{k+1}'

    #         # Inicializar la variable para almacenar la parte imaginaria
    #         y_IMAG = ""

    #         # Generar la parte imaginaria
    #         for k in range(N+1):
    #             for m in range(M+1):
    #                 y_IMAG += f' + ({ (a[m * N + k].imag):.8f})*(_vin[n-{m}])**{k+1}'

    #         file.write(f'vout(_vin) = sqrt(({y_REAL})**2 + ({y_IMAG})**2)')
    #         file.write(f'\n\nfase(_vin) = atan(real(({y_IMAG})/({y_REAL})))')

    # equations_to_file(a, N, M, 'eq.txt')

    # Predicciones con N y M elegidos
    y_pred, error_pred = modelos(x_original, y_original, a, b, h, N, M, error_metric, modelo)

    return time, x_original, y_original, y_pred, error_pred