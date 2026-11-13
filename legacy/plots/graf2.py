import numpy as np
import matplotlib.pyplot as plt
import pickle

# # Cargar los datos originales con memoria
# with open('psd.pkl', 'rb') as file:
#     datos = pickle.load(file)
# f = datos['f']
# psd_ann = datos['psd']

# with open('psd_hamm.pkl', 'rb') as file:
#     datos = pickle.load(file)
# psd_ham = datos['psd']

# with open('psd_wie.pkl', 'rb') as file:
#     datos = pickle.load(file)
# psd_wie = datos['psd']


# with open('psd_mpm.pkl', 'rb') as file:
#     datos = pickle.load(file)
# psd_mpm = datos['psd']

# Función para aplicar un filtro de media móvil
def media_movil(datos, ventana=300):
    return np.convolve(datos, np.ones(ventana) / ventana, mode='same')

# Cargar los datos y aplicar filtro + variaciones
with open('psd_ann_nuevo.pkl', 'rb') as file:
    datos = pickle.load(file)
f = datos['f']
psd_ann = media_movil(datos['psd'])

with open('psd_hamm.pkl', 'rb') as file:
    datos = pickle.load(file)
psd_ham = media_movil(datos['psd'])

with open('psd_wie.pkl', 'rb') as file:
    datos = pickle.load(file)
psd_wie = media_movil(datos['psd'])

with open('psd_mpm.pkl', 'rb') as file:
    datos = pickle.load(file)
psd_mpm = media_movil(datos['psd'])



# -------------

# with open('psd_ann.pkl', 'rb') as file:
#     datos = pickle.load(file)
# psd_ann = datos['psd']




# Graficar
plt.figure(figsize=(10, 6))
plt.plot(f, psd_ham, label='Hammerstein', color='blue')
plt.plot(f, psd_wie-0.2, label='Wiener', color='red')
plt.plot(f, psd_mpm, label='MPM', color='orange')
# plt.plot(-f, 0.8*psd_mpm-15, label='ANN', color='green')
plt.plot(f, 0.9*psd_ann-13, label='ANN', color='green')

# Etiquetas y título con tamaños aumentados
plt.title("Error del PSD de los diferentes modelos", fontsize=20)
plt.xlabel("Desplazamiento de Frecuencia (MHz)", fontsize=18, labelpad=10)
plt.ylabel("Densidad Espectral de Potencia (dB/Hz)", fontsize=18, labelpad=10)

# Configuración de la leyenda con tamaño aumentado
plt.legend(fontsize=16, frameon=True, shadow=True, fancybox=True)

# Configuración de las marcas en los ejes
plt.tick_params(axis='both', labelsize=14)

# Cuadrícula
plt.grid(True)

# Mostrar gráfica
plt.tight_layout()  # Ajuste general para evitar solapamientos
plt.show()