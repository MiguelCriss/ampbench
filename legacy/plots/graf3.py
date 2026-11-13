import numpy as np
import matplotlib.pyplot as plt
import pickle

# Configuración global para aumentar el tamaño de fuente
plt.rc('font', size=14)           # Tamaño de fuente general
plt.rc('axes', titlesize=16)      # Tamaño de los títulos de los ejes
plt.rc('axes', labelsize=14)      # Tamaño de las etiquetas de los ejes
plt.rc('xtick', labelsize=14)     # Tamaño de las etiquetas del eje X
plt.rc('ytick', labelsize=14)     # Tamaño de las etiquetas del eje Y
plt.rc('legend', fontsize=16)     # Tamaño de la leyenda
plt.rc('figure', titlesize=18)    # Tamaño del título de la figura

# Cargar los datos originales con memoria
with open('datos_salida.pkl', 'rb') as file:
    datos = pickle.load(file)
vout_orig = datos['vout_orig']
fase_orig = np.degrees(datos['fase_orig'])  # Convertir a grados

with open('datos_ham.pkl', 'rb') as file:
    datos = pickle.load(file)
vin = datos['vin']
vout_hamm = datos['vout']
fase_hamm = np.degrees(datos['fase'])

with open('datos_wie.pkl', 'rb') as file:
    datos = pickle.load(file)
vout_wie = datos['vout']
fase_wie = np.degrees(datos['fase'])

with open('datos_mpm.pkl', 'rb') as file:
    datos = pickle.load(file)
vout_mpm = datos['vout']
fase_mpm = np.degrees(datos['fase'])

with open('datos_ann.pkl', 'rb') as file:
    datos = pickle.load(file)
vout_ann = datos['vout']
fase_ann = np.degrees(datos['fase'])

# Crear subplots organizados
fig1, axes = plt.subplots(5, 2, figsize=(14, 20))
fig1.suptitle('Comparación del Voltaje de salida y la Fase', fontsize=20)

# Datos para Vout y Fase
vout_data = [
    (vout_orig, 'Original', 'blue'),
    (vout_hamm, 'Hammerstein', 'orange'),
    (vout_wie, 'Wiener', 'green'),
    (vout_mpm, 'MPM', 'red'),
    (vout_ann, 'ANN', 'purple')
]

fase_data = [
    (fase_orig, 'Original', 'blue'),
    (fase_hamm, 'Hammerstein', 'orange'),
    (fase_wie, 'Wiener', 'green'),
    (fase_mpm, 'MPM', 'red'),
    (fase_ann, 'ANN', 'purple')
]

# Llenar los subplots
for i in range(5):
    # Gráfico de Vout
    axes[i, 0].plot(vin, vout_data[i][0], label=vout_data[i][1], color=vout_data[i][2])
    axes[i, 0].set_ylabel('Vout (V)')
    axes[i, 0].grid(True)
    axes[i, 0].legend()

    # Gráfico de Fase
    axes[i, 1].plot(vin, fase_data[i][0], label=fase_data[i][1], color=fase_data[i][2])
    axes[i, 1].set_ylabel('Fase (°)')
    axes[i, 1].grid(True)
    axes[i, 1].legend()

# Etiquetas del eje X solo en las últimas gráficas
axes[4, 0].set_xlabel('Vin (V)')
axes[4, 1].set_xlabel('Vin (V)')

# Ajustar el espaciado entre subplots
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Mostrar la ventana de subplots
plt.show()

# # Crear la gráfica combinada
# fig2, ax_combined = plt.subplots(2, 1, figsize=(14, 12))
# fig2.suptitle('Gráfica Combinada de Vout y Fase', fontsize=20)

# # Gráfico combinado de Vout
# for data, label, color in vout_data:
#     ax_combined[0].plot(vin, data, label=f'Vout {label}', color=color)
# ax_combined[0].set_ylabel('Vout (V)')
# ax_combined[0].set_xlabel('Vin (V)')
# ax_combined[0].grid(True)
# ax_combined[0].legend()

# # Gráfico combinado de Fase
# for data, label, color in fase_data:
#     ax_combined[1].plot(vin, data, label=f'Fase {label}', color=color)
# ax_combined[1].set_ylabel('Fase (°)')
# ax_combined[1].set_xlabel('Vin (V)')
# ax_combined[1].grid(True)
# ax_combined[1].legend()

# # Ajustar el espaciado en la gráfica combinada
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# # Mostrar la ventana de gráfica combinada
# plt.show()
