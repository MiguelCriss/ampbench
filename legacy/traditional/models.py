import numpy as np
# import matplotlib.pyplot as plt
from funciones import *
from funciones_modelos import *

# Definir modelo (Opciones: 'Hammerstein', 'Wiener', 'MPM')
modelo = 'Wiener'
# Entrenar el modelo (Opciones: 'si', 'no')
entrenar = 'si'

# Definir rangos de valores para N y M
valores_N = list(range(0, 10 + 1))
valores_M = list(range(0, 10 + 1))
# Definir valores óptimos de N y M según las graficas de error
N = 1
M = 7
# Métrica de error a utilizar (puede ser 'mse', 'rmse', 'mae', 'mape', 'nmse', 'nmse_db' o 'medae')
error_metric = 'NMSE_dB'
# Tipo de datos (tipo=0 : vin,vout) (tipo=1 : I,Q)
tipo_datos = 1

# Entrenamiento, validación, prueba y predicciones
time, x_original, y_original, y_pred, error_pred = usar_modelo(tipo_datos, entrenar, valores_N, valores_M, N, M, error_metric, modelo)

# cálculos para gráficos
Vout_pred, fase_pred, Vin_original, Vout_original, fase_original = generar_predicciones(x_original, y_original, y_pred, error_metric, error_pred, f'predicciones_{modelo}.txt')

I_in = np.real(x_original)
Q_in = np.imag(x_original)

I_out_pred = np.real(y_pred)
Q_out_pred = np.imag(y_pred)

I_out_orig = np.real(y_original)
Q_out_orig = np.imag(y_original)



import pickle
# Guardar los datos en un archivo
with open('datos_salida.pkl', 'wb') as file:
    pickle.dump({'vin': Vin_original, 'vout': Vout_pred, 'fase': fase_pred, 'vout_orig': Vout_original, 'fase_orig': fase_original}, file)
print('OK')
# GRAFICAS

graficas(time, Vin_original, Vout_original, Vout_pred, fase_original, fase_pred, modelo, N, M,
          I_out_pred, Q_out_pred, I_out_orig, Q_out_orig)

