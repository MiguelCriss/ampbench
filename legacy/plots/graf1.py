import numpy as np
import matplotlib.pyplot as plt
import pickle

# Cargar los datos originales con memoria
with open('timeseries.pkl', 'rb') as file:
    datos = pickle.load(file)

# Acceder a los valores de tiempo y Vout
time = datos['t']
Vout_original = datos['vout_orig']
Vout_pred_con_M = datos['vout']

# Cargar los datos predichos sin memoria
with open('timeseries_sinM.pkl', 'rb') as file:
    datos_sinM = pickle.load(file)

Vout_pred_sin_M = datos_sinM['vout']

# Convertir a milivoltios (mV)
Vout_original_mV = np.array(Vout_original) * 1000
Vout_pred_conM_mV = np.array(Vout_pred_con_M) * 1000
Vout_pred_sin_M_mV = np.array(Vout_pred_sin_M) * 1000

# Configuración de estilo LaTeX y tamaño de fuente
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

# Crear la gráfica
plt.figure(figsize=(12, 6))

# Graficar los datos originales en línea azul continua
plt.plot(time, Vout_original_mV, 'b-', label=r'\textbf{Datos Originales}')

# Graficar las predicciones con memoria en puntos rojos
plt.plot(time, Vout_pred_conM_mV, 'ro', markersize=4, label=r'\textbf{Red Neuronal con Memoria}')

# Graficar las predicciones sin memoria en puntos verdes
plt.plot(time, Vout_pred_sin_M_mV, 'go', markersize=4, label=r'\textbf{Red Neuronal sin Memoria}')

# Configurar etiquetas y título
plt.xlabel(r'\textbf{Tiempo (s)}', fontsize=18)
plt.ylabel(r'\textbf{Vout (mV)}', fontsize=18)
plt.title(r'\textbf{Comparación de Vout Original y Predicciones de la Red Neuronal}', fontsize=20)

# Mostrar leyenda
plt.legend(fontsize=14)

# Añadir una cuadrícula para mejor visualización
plt.grid(True)

# Ajustar el layout para evitar superposiciones
plt.tight_layout()

# Mostrar la gráfica
plt.show()
