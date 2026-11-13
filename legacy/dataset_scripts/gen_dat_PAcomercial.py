import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

def plot_data(data, title):
    """
    Plotea los datos en dos subplots y muestra la gráfica en una ventana maximizada con estilo mejorado.
    
    Args:
    data (numpy.ndarray): Matriz de datos con al menos tres columnas.
    title (str): Título de la gráfica.
    """
    if data.shape[1] < 3:
        raise ValueError("Los datos deben tener al menos tres columnas.")

    vin = data[:, 0]
    vout = data[:, 1]
    fase = data[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(title, fontsize=25, fontweight='bold', color='#CC0000')

    # Colores personalizados
    color_puntos = '#0000CC'
    color_ejes = '#006600'
    color_numeros = '#FF3300'
    color_marco = '#000000'

    # Tamaño de los números de los ejes
    tamaño_numeros = 14

    # Función para formatear los números en los ejes
    def formatear_numeros(x, _):
        if x.is_integer():
            return f'{int(x)}'
        else:
            return f'{x:.2f}'

    # Primer subplot
    axes[0].scatter(vin, vout, alpha=0.6, edgecolors='w', linewidth=0.5, color=color_puntos)
    axes[0].set_xlabel('vin', fontsize=23, color=color_ejes)
    axes[0].set_ylabel('vout', fontsize=23, color=color_ejes)
    axes[0].set_title('vin vs vout', fontsize=20, color='#CC0000')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(formatear_numeros))
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(formatear_numeros))

    # Segundo subplot
    axes[1].scatter(vin, fase, alpha=0.6, edgecolors='w', linewidth=0.5, color=color_puntos)
    axes[1].set_xlabel('vin', fontsize=23, color=color_ejes)
    axes[1].set_ylabel('fase', fontsize=23, color=color_ejes)
    axes[1].set_title('vin vs fase', fontsize=20, color='#CC0000')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(formatear_numeros))
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(formatear_numeros))

    # Ajustar límites y formato de ejes
    for ax in axes:
        ax.spines['top'].set_color(color_marco)
        ax.spines['right'].set_color(color_marco)
        ax.spines['left'].set_color(color_marco)
        ax.spines['bottom'].set_color(color_marco)
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=tamaño_numeros, labelcolor=color_numeros)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Leer los datos originales
original_data = np.genfromtxt('dataset/dataset.txt')

# Asegurarse de que los datos están en el formato correcto
assert original_data.shape[1] == 3

# Extraer las columnas
vin_original = original_data[:, 0]
vout_original = original_data[:, 1]
fase_original = original_data[:, 2]

# Crear funciones de interpolación
interp_vout = interp1d(vin_original, vout_original, kind='linear', fill_value="extrapolate")
interp_fase = interp1d(vin_original, fase_original, kind='linear', fill_value="extrapolate")

# Generar valores de vin
num_generated_rows = 10000
vin_generated = np.linspace(np.min(vin_original), np.max(vin_original), num_generated_rows)

# Interpolar para obtener vout y fase
vout_generated = interp_vout(vin_generated)
fase_generated = interp_fase(vin_generated)

# Añadir ruido
std_devs = np.std(original_data, axis=0)
noise_vout = np.random.normal(scale=0.03 * std_devs[1], size=num_generated_rows) #0.03
noise_fase = np.random.normal(scale=0.03 * std_devs[2], size=num_generated_rows)

vout_generated += noise_vout
fase_generated += noise_fase

# Generar datos
generated_data = np.column_stack((vin_generated, vout_generated, fase_generated))

# Dividir los datos generados en conjuntos de train (70%), test (15%) y validation (15%)
np.random.shuffle(generated_data)
total_rows = generated_data.shape[0]
train_data = generated_data[:int(0.7 * total_rows)]
test_data = generated_data[int(0.7 * total_rows):int(0.85 * total_rows)]
val_data = generated_data[int(0.85 * total_rows):]

# Guardar los conjuntos de datos en archivos separados
np.savetxt('dataset/train.txt', train_data, fmt='%.8f')
np.savetxt('dataset/test.txt', test_data, fmt='%.8f')
np.savetxt('dataset/valid.txt', val_data, fmt='%.8f')

# Leer los datos generados
generated_data = np.genfromtxt('dataset/dataset.txt') 
generated_train = np.genfromtxt('dataset/train.txt') 
generated_valid = np.genfromtxt('dataset/valid.txt') 
generated_test = np.genfromtxt('dataset/test.txt') 

# Llamar a la función con un título
plot_data(generated_data, 'Datos Originales')
plot_data(generated_train, 'Datos Generados para el entrenamiento')
plot_data(generated_valid, 'Datos Generados para la validación')
plot_data(generated_test, 'Datos Generados para la prueba')
