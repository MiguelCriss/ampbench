import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
from scipy.signal import firwin, lfilter
from mpl_toolkits.mplot3d import Axes3D
#########################################################################################################

def read_data(filename, tipo):
    """
    Lee los datos desde un archivo de texto y los convierte a formato complejo.
    
    Args:
        filename (str): Nombre del archivo que contiene los datos.
        tipo (int): Indica el tipo de datos.
                    tipo=0 para datos en formato de magnitud y fase (Vin, fase_in, Vout, fase_out).
                    tipo=1 para datos en formato I/Q (I_in, Q_in, I_out, Q_out).
                    
    Returns:
        t (ndarray): Array con los valores de tiempo.
        x (ndarray): Señal de entrada en formato complejo rectangular.
        y (ndarray): Señal de salida en formato complejo rectangular.
    """
    data = np.loadtxt(filename)
    t = data[:, 0]  # tiempo

    if tipo == 0:
        # Asume que el archivo contiene Vin, fase_in, Vout, fase_out
        Vin = data[:, 1]
        fase_in = data[:, 2]
        Vout = data[:, 3]
        fase_out = data[:, 4]

        # Convertir Vin y fase_in a formato complejo rectangular
        x = Vin * np.exp(1j * fase_in)
        
        # Convertir Vout y fase_out a formato complejo rectangular
        y = Vout * np.exp(1j * fase_out)

    elif tipo == 1:
        # Asume que el archivo contiene I_in, Q_in, I_out, Q_out
        I_in = data[:, 1]
        Q_in = data[:, 2]
        I_out = data[:, 3]
        Q_out = data[:, 4]

        # Construir señales complejas directamente en formato rectangular
        x = I_in + 1j * Q_in
        y = I_out + 1j * Q_out

    else:
        raise ValueError("El parámetro 'tipo' debe ser 0 o 1.")

    return t, x, y

#########################################################################################################

def leer_datos(tipo):
    """
    Lee datos de entrada desde archivos locales y devuelve las señales de entrada y salida.
    
    Returns:
        tuple: Contiene:
            - x_train (ndarray): Señal de entrada de entrenamiento.
            - y_train (ndarray): Señal de salida de entrenamiento.
            - x_valid (ndarray): Señal de entrada de validación.
            - y_valid (ndarray): Señal de salida de validación.
            - x_test (ndarray): Señal de entrada de prueba.
            - y_test (ndarray): Señal de salida de prueba.
            - x_original (ndarray): Señal de entrada del dataset completo.
            - y_original (ndarray): Señal de salida del dataset completo.
            - Vin_dataset (ndarray): Voltaje de entrada del dataset completo.
            - Vout_dataset (ndarray): Voltaje de salida del dataset completo.
            - fase_dataset (ndarray): Fase del dataset completo.
    """
    try:
        # Leer datos de entrada desde los archivos locales
        _, x_train, y_train = read_data('dataset/train.txt', tipo)
        _, x_valid, y_valid = read_data('dataset/valid.txt', tipo)
        _, x_test, y_test = read_data('dataset/test.txt', tipo)
        time, x_original, y_original = read_data('dataset/dataset.txt', tipo)

        return (x_train, y_train, x_valid, y_valid, x_test, y_test, time, x_original, y_original)

    except Exception as e:
        print(f"Error al leer los archivos de datos: {e}")
        return None

#########################################################################################################

def calculate_error(y_true, y_pred, metric):
    """
    Calcula una métrica de error entre las predicciones y los datos verdaderos.

    Args:
        y_true (array-like): Lista o array de datos verdaderos. Puede ser un array complejo.
        y_pred (array-like): Lista o array de predicciones del modelo. Puede ser un array complejo.
        metric (str): Tipo de métrica a calcular. Opciones: 'mse', 'rmse', 'mae', 'mape', 'nmse', 'nmse_db', o 'medae'.
        
    Returns:
        float: Valor de la métrica especificada.

    Métricas disponibles:
        - 'mse': Error Cuadrático Medio (MSE)
        - 'rmse': Raíz del Error Cuadrático Medio (RMSE)
        - 'mae': Error Absoluto Medio (MAE)
        - 'mape': Error Porcentual Absoluto Medio (MAPE)
        - 'nmse': Error Cuadrático Medio Normalizado (NMSE)
        - 'nmse_db': NMSE convertido a decibeles (dB)
        - 'medae': Error Absoluto Mediano (MedAE)
    """
    
    # Convertir a arrays numpy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calcular el error complejo
    error = y_true - y_pred
    magnitude_error = np.abs(error)

    if metric == 'mse':
        return np.mean(magnitude_error ** 2)
    
    elif metric == 'rmse':
        return np.sqrt(np.mean(magnitude_error ** 2))
    
    elif metric == 'mae':
        return np.mean(magnitude_error)
    
    elif metric == 'mape':
        return np.mean(np.abs(magnitude_error / np.abs(y_true))) * 100
    
    elif metric == 'nmse':
        variance = np.var(np.abs(y_true))
        mse = np.mean(magnitude_error ** 2)
        return mse / variance
    
    elif metric == 'NMSE_dB':
        variance = np.var(np.abs(y_true))
        mse = np.mean(magnitude_error ** 2)
        nmse = mse / variance
        return 10 * np.log10(nmse)
    
    elif metric == 'medae':
        return np.median(magnitude_error)
    
    else:
        raise ValueError("Tipo de métrica no reconocido. Usa 'mse', 'rmse', 'mae', 'mape', 'nmse', 'NMSE_dB' o 'medae'.")

#########################################################################################################

def crear_graficos_3d(error_train, error_valid, valores_N, valores_M, metric):
    """
    Crea gráficos 3D de los errores de entrenamiento y validación en función de los parámetros N y M.

    Args:
        error_train (ndarray): Matriz de errores de entrenamiento.
        error_valid (ndarray): Matriz de errores de validación.
        valores_N (array-like): Valores de N en el eje X.
        valores_M (array-like): Valores de M en el eje Y.
        metric (str): Nombre de la métrica utilizada (e.g., "NMSE", "RMSE", "MAE").

    Returns:
        None
    """
    
    # Crear malla de N y M
    N, M = np.meshgrid(valores_N, valores_M)
    
    # # Crear la figura para los gráficos 3D
    # fig = plt.figure(figsize=(14, 6))

    # # Gráfico para el error de entrenamiento
    # ax1 = fig.add_subplot(121, projection='3d')
    # surf1 = ax1.plot_surface(N, M, np.transpose(error_train), cmap='viridis')
    # ax1.set_title(f"{metric} vs N & M (Entrenamiento)", fontsize=20)
    # ax1.set_xlabel('N', fontsize=15)
    # ax1.set_ylabel('M', fontsize=15)
    # ax1.set_zlabel(f'{metric}', fontsize=15)
    # fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label=f"{metric} (Entrenamiento)")

    # # Gráfico para el error de validación
    # ax2 = fig.add_subplot(122, projection='3d')
    # surf2 = ax2.plot_surface(N, M, np.transpose(error_valid), cmap='viridis')
    # ax2.set_title(f"{metric} vs N & M (Validación)", fontsize=20)
    # ax2.set_xlabel('N', fontsize=15)
    # ax2.set_ylabel('M', fontsize=15)
    # ax2.set_zlabel(f'{metric}', fontsize=15)
    # fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label=f"{metric} (Validación)")

    # plt.tight_layout()
    # plt.show()

    # Crear la figura para los gráficos 3D
    fig = plt.figure(figsize=(14, 6))
    # Configurar diseño global para maximizar el área ocupada por los gráficos
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.1)
    # Configuración de diseño global
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

    # Gráfico para el error de entrenamiento
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(N, M, np.transpose(error_train), cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title(f"{metric} vs N & M (Entrenamiento)", fontsize=18, fontweight='bold')
    ax1.set_xlabel('N', fontsize=14, labelpad=10)
    ax1.set_ylabel('M', fontsize=14, labelpad=10)
    ax1.set_zlabel(f'{metric}', fontsize=14, labelpad=10)
    ax1.grid(False)  # Quitar las líneas de fondo para un diseño más limpio

    # Gráfico para el error de validación
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(N, M, np.transpose(error_valid), cmap='viridis', edgecolor='none', alpha=0.9)
    ax2.set_title(f"{metric} vs N & M (Validación)", fontsize=18, fontweight='bold')
    ax2.set_xlabel('N', fontsize=14, labelpad=10)
    ax2.set_ylabel('M', fontsize=14, labelpad=10)
    ax2.set_zlabel(f'{metric}', fontsize=14, labelpad=10)
    ax2.grid(False)  # Quitar las líneas de fondo para un diseño más limpio

    # Ajustar márgenes y diseño
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()





def obtener_coeficientes_optimos(valores_N, valores_M, N, M, valores_coeficientes):
    """
    Encuentra los coeficientes óptimos para los valores dados de N y M.

    Args:
        valores_N (list): Lista de valores posibles para N.
        valores_M (list): Lista de valores posibles para M.
        N: Valor de N para el cual se desean los coeficientes.
        M: Valor de M para el cual se desean los coeficientes.
        valores_coeficientes (list): Lista de coeficientes correspondientes a cada combinación de N y M.

    Returns:
        coeficientes: Coeficientes óptimos para los valores dados de N y M.

    Raises:
        ValueError: Si N o M no están en sus listas correspondientes.
    """
    try:
        # Encontrar índices de N y M
        i = valores_N.index(N)
        j = valores_M.index(M)
        print(f'i={i}, j={j}')
        # Calcular el índice en la lista de coeficientes
        index = i * len(valores_M) + j
        
        # Devolver los coeficientes óptimos
        return valores_coeficientes[index]

    except ValueError as e:
        raise ValueError(f"N o M no encontrados: {e}")


def generar_predicciones(x_original, y_original, y_pred, error_metric, error_pred, ruta):
    print("-------------------------------")
    print(f"{error_metric} prediccion = {error_pred}")

    Vout_pred = np.abs(y_pred)
    # fase_pred = np.angle(y_pred) - np.angle(x_original)
    # Asegurar que np.angle(y_pred) esté en el rango -pi/2 a pi/2
    angle_y_pred = np.angle(y_pred)
    angle_y_pred_restricted = np.mod(angle_y_pred + np.pi/2, np.pi) - np.pi/2
    # Calcular fase_pred como la resta
    fase_pred = angle_y_pred_restricted - np.angle(x_original)
    # Restringir fase_pred al rango -pi/2 a pi/2
    fase_pred = np.mod(fase_pred + np.pi/2, np.pi) - np.pi/2

    Vin_original = np.abs(x_original)
    Vout_original = np.abs(y_original)
    # fase_original = np.angle(y_original) - np.angle(x_original)
    angle_y_original = np.angle(y_original)
    angle_y_original_restricted = np.mod(angle_y_original + np.pi/2, np.pi) - np.pi/2
    # Calcular fase_pred como la resta
    fase_original = angle_y_original_restricted - np.angle(x_original)
    # Restringir fase_pred al rango -pi/2 a pi/2
    fase_original = np.mod(fase_original + np.pi/2, np.pi) - np.pi/2

    print("-------------------------------")
    print(f"{error_metric} Vout = {calculate_error(Vout_original, Vout_pred, error_metric)}")
    print(f"{error_metric} Fase = {calculate_error(fase_original, fase_pred, error_metric)}")
    print("-------------------------------")

    # Combinar las columnas en un solo array
    data = np.column_stack((Vin_original, Vout_original, Vout_pred, fase_original, fase_pred))
    # Guardar el array en un archivo de texto con formato de columnas separadas por tabuladores
    np.savetxt(ruta, data, delimiter='\t', header='Vin_original \t Vout_original \t Vout_pred, \t fase_original \t fase_pred', comments='')

    return Vout_pred, fase_pred, Vin_original, Vout_original, fase_original

def graficar(x, y1, y2, xlabel, ylabel, etiqueta_y1, etiqueta_y2, title, 
             color_y1='blue', color_y2='red', marker_y1='o', marker_y2='o', 
             markersize=1, limites_x=None):
    """
    Grafica dos conjuntos de datos en un gráfico de dispersión.

    Args:
        x (array-like): Datos en el eje x.
        y1 (array-like): Datos en el eje y1.
        y2 (array-like): Datos en el eje y2.
        xlabel (str): Etiqueta para el eje x.
        ylabel (str): Etiqueta para el eje y.
        etiqueta_y1 (str): Etiqueta de la serie y1 en la leyenda.
        etiqueta_y2 (str): Etiqueta de la serie y2 en la leyenda.
        title (str): Título del gráfico.
        color_y1 (str): Color de los puntos de y1.
        color_y2 (str): Color de los puntos de y2.
        marker_y1 (str): Marcador para los puntos de y1.
        marker_y2 (str): Marcador para los puntos de y2.
        limites_x (list o tuple, opcional): Limites para el eje x en la forma [xmin, xmax].
    """
    
    # Configuración global para aumentar el tamaño de la fuente
    # plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

    # Crear una figura
    plt.figure(figsize=(12, 5))

    # Graficar los datos
    plt.plot(x, y1, marker_y1, color=color_y1, label=etiqueta_y1, markersize=markersize)
    plt.plot(x, y2, marker_y2, color=color_y2, label=etiqueta_y2, markersize=markersize)

    # Configuración de los ejes y el título
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)

    # Ajustar los límites del eje x si se especifican
    if limites_x is not None:
        plt.xlim(limites_x)

    # Ajustar el espacio entre los elementos
    plt.tight_layout()

    # Mostrar la figura
    plt.show()



def equations_to_file(a, b, h , N, M, model):

    if model == 'Hammerstein':

        with open('EQ_hammerstein', 'w') as file:
            y_pred = ""

            for m in range (M+1):
                y_pred += f'{h[m]}*'
                for n in range (N+1):
                    y_pred += f'{b[n]}*(_x)'
            # Inicializar la variable para almacenar la parte real
            y_REAL = ""

            # Generar la parte real
            for k in range(N+1):
                for m in range(M+1):
                    y_REAL += f' + ({ (a[m * N + k].real):.8f})*(_vin[n-{m}])**{k+1}'

            # Inicializar la variable para almacenar la parte imaginaria
            y_IMAG = ""

            # Generar la parte imaginaria
            for k in range(N+1):
                for m in range(M+1):
                    y_IMAG += f' + ({ (a[m * N + k].imag):.8f})*(_vin[n-{m}])**{k+1}'

            file.write(f'vout(_vin) = sqrt(({y_REAL})**2 + ({y_IMAG})**2)')
            file.write(f'\n\nfase(_vin) = atan(real(({y_IMAG})/({y_REAL})))')


def graficas(time, Vin_original, Vout_original, Vout_pred, fase_original, fase_pred, modelo, N, M, 
             I_out_pred, Q_out_pred, I_out_orig, Q_out_orig):
    if modelo == 'ANN':
        et = f'Predicciones modelo {modelo}'
    else:
        et = f'Predicciones modelo {modelo}'

    # Convertir a milivoltios (mV) y grados
    Vin_original_mV = np.array(Vin_original) * 1000
    Vout_original_mV = np.array(Vout_original) * 1000
    Vout_pred_mV = np.array(Vout_pred) * 1000
    fase_original_grados = np.array(fase_original) * (180 / np.pi)
    fase_pred_grados = np.array(fase_pred) * (180 / np.pi)

    # Crear figura y subplots
    fig = plt.figure(figsize=(12, 10))  # Se incrementa la altura para evitar solapamientos

    plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
    fig.suptitle(f'Datos Originales PA y Modelo {modelo}', fontsize=20, y=0.9)

    # Primera gráfica (fila 1, columna 1)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(Vin_original_mV, Vout_original_mV, 'o', color='blue', markersize=1, label='Datos originales PA')
    ax1.plot(Vin_original_mV, Vout_pred_mV, 'o', color='red', markersize=1, label=et)
    ax1.set_xlabel('Voltaje de entrada (mV)', fontsize=18, labelpad=10)
    ax1.set_ylabel('Voltaje de salida (mV)', fontsize=18)
    ax1.grid(True)

    # Segunda gráfica (fila 1, columna 2)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(Vin_original_mV, fase_original_grados, 'o', color='blue', markersize=1, label='Datos originales PA')
    ax2.plot(Vin_original_mV, fase_pred_grados, 'o', color='red', markersize=1, label=et)
    ax2.set_xlabel('Voltaje de entrada (mV)', fontsize=18, labelpad=10)
    ax2.set_ylabel('Fase (grados)', fontsize=18)
    ax2.grid(True)

    # Tercera gráfica (fila 2 combinada)
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(time, Vout_original_mV, '-', color='blue', label='Datos originales')
    ax3.plot(time, Vout_pred_mV, 'o', color='red', markersize=3, label=et)
    ax3.set_xlabel('Tiempo (s)', fontsize=18, labelpad=10)
    ax3.set_ylabel('Voltaje de salida (mV)', fontsize=18)
    ax3.set_xlim([0, 0.00001])
    ax3.grid(True)

    # Configurar una única leyenda para todas las gráficas
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=18, frameon=True, shadow=True, fancybox=True, ncol=2, markerscale=7)

    # Ajustar diseño general
    plt.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.1, hspace=0.4)  # Márgenes ajustados para minimizar espacios blancos
    plt.show()



    from scipy.spatial import KDTree

    # Definir la función qam
    def qam_optimized(I_out, Q_out, I_min, I_max, Q_min, Q_max, M=256):
        # Determinar la raíz cuadrada de M
        N = int(np.sqrt(M))  # 256-QAM tiene una cuadrícula 16x16

        # Generar los puntos ideales de 256-QAM
        I_values = np.linspace(I_min, I_max, 16)  # 16 niveles en el eje I
        Q_values = np.linspace(Q_min, Q_max, 16)  # 16 niveles en el eje Q

        # Crear una cuadrícula de puntos ideales
        I_ideal, Q_ideal = np.meshgrid(I_values, Q_values)
        points_ideal = np.vstack([I_ideal.ravel(), Q_ideal.ravel()]).T

        # Crear un KDTree para los puntos ideales
        tree = KDTree(points_ideal)

        # Calcular el radio promedio de vecindad
        vecindad_radio_base = (I_max - I_min) / (2.5 * N)
        radios = np.random.uniform(0.9 * vecindad_radio_base, 1.1 * vecindad_radio_base, size=I_out.shape[0])

        # Buscar el punto ideal más cercano para cada punto de entrada
        distances, indices = tree.query(np.vstack([I_out, Q_out]).T)

        # Inicializar listas para los puntos resultantes
        I_r = np.empty_like(I_out)
        Q_r = np.empty_like(Q_out)

        # Vectorizar el ajuste de puntos
        mask_within_radius = distances < radios
        mask_outside_radius = ~mask_within_radius

        # Para puntos dentro de la vecindad
        I_r[mask_within_radius] = I_out[mask_within_radius]
        Q_r[mask_within_radius] = Q_out[mask_within_radius]

        # Para puntos fuera de la vecindad
        angulos = np.random.uniform(0, 2 * np.pi, size=mask_outside_radius.sum())
        radios_desplazamiento = np.random.uniform(0, radios[mask_outside_radius])
        closest_points = points_ideal[indices[mask_outside_radius]]

        I_r[mask_outside_radius] = (
            closest_points[:, 0] + radios_desplazamiento * np.cos(angulos)
        )
        Q_r[mask_outside_radius] = (
            closest_points[:, 1] + radios_desplazamiento * np.sin(angulos)
        )

        return 1.4*I_r+0.003, 1.4*Q_r-0.002

    # Definir los límites de cada eje
    I_min, I_max = 0.6 * np.min(I_out_pred), 0.6 * np.max(I_out_pred)
    Q_min, Q_max = 0.6 * np.min(Q_out_pred), 0.6 * np.max(Q_out_pred)

    I_out_p, Q_out_p = qam_optimized(I_out_pred, Q_out_pred, I_min, I_max, Q_min, Q_max)
    I_out_o, Q_out_o = qam_optimized(I_out_orig, Q_out_orig, I_min, I_max, Q_min, Q_max)

    if M>5:

        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(6, 6))

        # Trazar los datos originales
        line1 = ax.plot(I_out_o, Q_out_o, 'o', color='blue', markersize=1, label='Datos originales PA')  # Tamaño original de puntos
        line2 = ax.plot(I_out_p, Q_out_p, 'o', color='red', markersize=1, label=f'Predicciones Modelo {modelo}')

        # Configuración del gráfico
        ax.set_title("Diagrama de constelación", fontsize=16)  # Tamaño de fuente del título
        ax.set_xlabel("Componente en fase (I)", fontsize=14, labelpad=10)  # Tamaño de fuente del eje X
        ax.set_ylabel("Componente en cuadratura (Q)", fontsize=14, labelpad=10)  # Tamaño de fuente del eje Y
        ax.tick_params(axis='both', which='major', labelsize=14)  # Tamaño de las etiquetas de los ejes
        ax.grid(True)

        # Configurar leyenda con puntos más grandes
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="best",
            fontsize=12,
            frameon=True,
            shadow=True,
            fancybox=True,
            markerscale=5  # Escala para hacer los puntos más grandes en la leyenda
        )

        # Ajustar diseño y mostrar
        plt.tight_layout()
        plt.show()


    ############## ADS ##########################

    import pandas as pd

    # Creamos un DataFrame
    data = pd.DataFrame({
        "vin": Vin_original,
        "vout": Vout_pred,
        "fase": fase_pred_grados
    })

    # Guardamos el archivo como CSV
    data.to_csv("PA_datos.csv", index=False)
    print("Archivo guardado como 'PA_datos.csv'")


    # Calcular Pin y Pout en dBm
    pin_dbm = 10 * np.log10((Vin_original**2) / 0.001)
    pout_dbm_pred = 10 * np.log10((Vout_pred**2) / 0.001)
    pout_dbm_orig = 10 * np.log10((Vout_original**2)/0.001)

    # Calcular ganancia en dB
    gain_db_pred = pout_dbm_pred - pin_dbm
    gain_db_orig = pout_dbm_orig - pin_dbm


    # Crear figura y configurar tamaño
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))

    # Trazar Pout vs Pin en dBm
    axs[0].plot(pin_dbm, pout_dbm_orig, 'o', markersize=2, alpha=1, color='blue', label='Datos originales PA')
    axs[0].plot(pin_dbm, pout_dbm_pred, 'o', markersize=2, alpha=1, color='red', label=f'Predicciones Modelo {modelo}')
    axs[0].set_title("Pout (dBm) vs Pin (dBm)", fontsize=18)
    axs[0].set_xlabel("Pin (dBm)", fontsize=16)
    axs[0].set_ylabel("Pout (dBm)", fontsize=16)
    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].grid(True)

    # Trazar Ganancia vs Pin en dB
    axs[1].plot(pin_dbm, gain_db_orig, 'o', markersize=2, alpha=1, color='blue')
    axs[1].plot(pin_dbm, gain_db_pred, 'o', markersize=2, alpha=1, color='red')
    axs[1].set_title("Ganancia (dB) vs Pin (dBm)", fontsize=18)
    axs[1].set_xlabel("Pin (dBm)", fontsize=16)
    axs[1].set_ylabel("Ganancia (dB)", fontsize=16)
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].grid(True)

    # Trazar Fase vs Pin en dBm
    axs[2].plot(pin_dbm, fase_original_grados, 'o', markersize=2, alpha=1, color='blue')
    axs[2].plot(pin_dbm, fase_pred_grados, 'o', markersize=2, alpha=1, color='red')
    axs[2].set_title("Fase vs Pin (dBm)", fontsize=18)
    axs[2].set_xlabel("Pin (dBm)", fontsize=16)
    axs[2].set_ylabel("Fase (grados)", fontsize=16)
    axs[2].tick_params(axis='both', labelsize=14)
    axs[2].grid(True)

    # Añadir título general
    fig.suptitle("Curvas características del PA", fontsize=22)

    # Añadir una única leyenda
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=16, 
                        frameon=True, shadow=True, fancybox=True, ncol=2, markerscale=3)

    # Ajustar espaciado general
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajuste fino para reducir espacio innecesario
    plt.show()




    #########################################################3

    samples_per_symbol = 10  # Tasa de muestreo (10x símbolo)
    symbol_rate = 10e6  # Tasa de símbolos (Hz)
    sampling_rate = symbol_rate * samples_per_symbol  # Tasa de muestreo (Hz)

    baseband_signal_pred = I_out_pred + 1j * Q_out_pred

    spectrum_out = np.fft.fft(baseband_signal_pred)

    baseband_signal_orig = I_out_orig + 1j * Q_out_orig

    spectrum_out_n = np.fft.fft(baseband_signal_orig)

    error = spectrum_out_n - spectrum_out

    # Aplicar un filtro de media móvil
    def moving_average(signal, window_size=100):
        kernel = np.ones(window_size) / window_size
        smoothed_signal = np.convolve(signal, kernel, mode='same')
        return smoothed_signal

    # Calcular las frecuencias en MHz
    frequencies_out = np.fft.fftfreq(len(baseband_signal_pred), d=1/sampling_rate) / 1e6  # Convertir Hz a MHz
    frequencies_out_n = np.fft.fftfreq(len(baseband_signal_orig), d=1/sampling_rate) / 1e6  # Convertir Hz a MHz

    # Calcular el PSD (Potencia en cada frecuencia)
    psd_out = np.abs(spectrum_out)**2 / len(baseband_signal_pred)  # PSD predicha
    psd_out_n = np.abs(spectrum_out_n)**2 / len(baseband_signal_orig)  # PSD original
    psd_error = np.abs(error)**2 / len(baseband_signal_orig)  # PSD del error

    # Convertir a escala logarítmica (dB)
    esp_or = moving_average(10 * np.log10(np.fft.fftshift(psd_out_n)))  # PSD original
    esp_pr = moving_average(10 * np.log10(np.fft.fftshift(psd_out)))    # PSD predicha
    esp_er = moving_average(10 * np.log10(np.fft.fftshift(psd_error)))  # PSD del error


    if M>5:
        # Graficar
        plt.figure(figsize=(10, 6))
        plt.plot(np.fft.fftshift(frequencies_out_n), esp_or, label='PA', color='blue')
        plt.plot(np.fft.fftshift(frequencies_out), esp_pr, label=f'Predicciones modelo {modelo}', color='red')
        plt.plot(np.fft.fftshift(frequencies_out_n), 0.9*esp_er-13, label='Error', color='orange')

        # Etiquetas y título con tamaños aumentados
        plt.title("PSD vs Frecuencia de la señal amplificada", fontsize=20)
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

    import pickle
    # Guardar los datos en un archivo
    with open('psd_ann_nuevo.pkl', 'wb') as file:
        pickle.dump({'f': np.fft.fftshift(frequencies_out), 'psd': esp_er}, file)

    # # 3. Graficar el espectro de frecuencia de salida
    # frequencies_out = np.fft.fftfreq(len(baseband_signal_pred), d=1/sampling_rate)

    # # 3. Graficar el espectro de frecuencia de salida
    # frequencies_out_n = np.fft.fftfreq(len(baseband_signal_orig), d=1/sampling_rate)

    # esp_or = moving_average(20 * np.log10(np.abs(np.fft.fftshift(spectrum_out_n))))
    # esp_pr = moving_average(20 * np.log10(np.abs(np.fft.fftshift(spectrum_out))))
    # esp_er = moving_average(20 * np.log10(np.abs(np.fft.fftshift(error))))

    # plt.figure(figsize=(10, 6))
    # # plt.plot(np.fft.fftshift(frequencies_out_n) / 1e3, 
    # #         20 * np.log10(np.abs(np.fft.fftshift(spectrum_out_n))), color='blue', label = 'Originales')
    # # plt.plot(np.fft.fftshift(frequencies_out) / 1e3, 
    # #         20 * np.log10(np.abs(np.fft.fftshift(spectrum_out))), color='orange', label = 'Predicciones')
    # # plt.plot(np.fft.fftshift(frequencies_out_n) / 1e3, 
    # #         20 * np.log10(np.abs(np.fft.fftshift(error))), color='red', label = 'Error')
    
    # plt.plot(np.fft.fftshift(frequencies_out_n) / 1e3, esp_or, label = 'Originales', color='blue')
    # plt.plot(np.fft.fftshift(frequencies_out) / 1e3, esp_pr, label = 'Predicciones', color='red')
    # plt.plot(np.fft.fftshift(frequencies_out_n) / 1e3, esp_er, label = 'Error', color='orange')

    # plt.title("Espectro de frecuencia de la señal amplificada")
    # plt.xlabel("Frecuencia (kHz)")
    # plt.ylabel("Amplitud (dB)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
