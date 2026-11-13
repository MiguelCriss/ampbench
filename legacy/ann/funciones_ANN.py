import numpy as np
import tensorflow as tf


def calcular_min_max(variable):
    valor_min, valor_max = min(variable), max(variable)
    return valor_min, valor_max

def normalizar(variable, valor_min, valor_max):
    rango = valor_max - valor_min
    return [(x - valor_min) / rango for x in variable]

def desnormalizar(variable_normalizada, valor_min, valor_max):
    rango = valor_max - valor_min
    return [(x * rango) + valor_min for x in variable_normalizada]



def leer(filename, tipo):
    data = np.loadtxt(filename)
    t = data[:, 0]  # tiempo

    if tipo == 0:
        # Asume que el archivo contiene Vin, fase_in, Vout, fase_out
        Vin = data[:, 1]
        fase_in = data[:, 2]
        Vout = data[:, 3]
        fase_out = data[:, 4]

        return t, Vin, fase_in, Vout, fase_out

    elif tipo == 1:
        # Asume que el archivo contiene I_in, Q_in, I_out, Q_out
        I_in = data[:, 1]
        Q_in = data[:, 2]
        I_out = data[:, 3]
        Q_out = data[:, 4]

        return t, I_in, Q_in, I_out, Q_out
    



def generar_entradas_salidas(tipo_datos, M):
    # Leer datos desde los archivos
    _, x1_train, x2_train, y1_train, y2_train = leer('dataset/train.txt', tipo_datos)
    _, x1_valid, x2_valid, y1_valid, y2_valid = leer('dataset/valid.txt', tipo_datos)
    _, x1_test, x2_test, y1_test, y2_test = leer('dataset/test.txt', tipo_datos)
    time, x1_original, x2_original, y1_original, y2_original = leer('dataset/dataset.txt', tipo_datos)

    # Manejo de valores de x2 iguales a cero
    def procesar_ceros(array):
        return np.where(array == 0, 0, array)

    x2_train = procesar_ceros(x2_train)
    x2_valid = procesar_ceros(x2_valid)
    x2_test = procesar_ceros(x2_test)
    x2_original = procesar_ceros(x2_original)

    # Identificar máximos y mínimos del conjunto de entrenamiento
    x1_min, x1_max = calcular_min_max(x1_train)
    x2_min, x2_max = calcular_min_max(x2_train)
    y1_min, y1_max = calcular_min_max(y1_train)
    y2_min, y2_max = calcular_min_max(y2_train)

    # Normalizar datos
    def normalizar_ceros(array, minimo, maximo):
        return np.where(array == 0, 0, normalizar(array, minimo, maximo))

    x1_train_N = normalizar(x1_train, x1_min, x1_max)
    x2_train_N = normalizar_ceros(x2_train, x2_min, x2_max)
    y1_train_N = normalizar(y1_train, y1_min, y1_max)
    y2_train_N = normalizar(y2_train, y2_min, y2_max)

    x1_valid_N = normalizar(x1_valid, x1_min, x1_max)
    x2_valid_N = normalizar_ceros(x2_valid, x2_min, x2_max)
    y1_valid_N = normalizar(y1_valid, y1_min, y1_max)
    y2_valid_N = normalizar(y2_valid, y2_min, y2_max)

    x1_test_N = normalizar(x1_test, x1_min, x1_max)
    x2_test_N = normalizar_ceros(x2_test, x2_min, x2_max)
    y1_test_N = normalizar(y1_test, y1_min, y1_max)
    y2_test_N = normalizar(y2_test, y2_min, y2_max)

    # Crear matrices de memoria
    x1_train_memory = create_memory_matrix(x1_train_N, M)
    x2_train_memory = create_memory_matrix(x2_train_N, M)
    x1_valid_memory = create_memory_matrix(x1_valid_N, M)
    x2_valid_memory = create_memory_matrix(x2_valid_N, M)
    x1_test_memory = create_memory_matrix(x1_test_N, M)
    x2_test_memory = create_memory_matrix(x2_test_N, M)

    # Combinar entradas
    X_train = np.hstack((x1_train_memory, x2_train_memory))
    X_valid = np.hstack((x1_valid_memory, x2_valid_memory))
    X_test = np.hstack((x1_test_memory, x2_test_memory))

    # Combinar salidas
    Y_train = np.hstack((np.array(y1_train_N).reshape(-1, 1), np.array(y2_train_N).reshape(-1, 1)))
    Y_valid = np.hstack((np.array(y1_valid_N).reshape(-1, 1), np.array(y2_valid_N).reshape(-1, 1)))
    Y_test = np.hstack((np.array(y1_test_N).reshape(-1, 1), np.array(y2_test_N).reshape(-1, 1)))

    # Procesar datos originales
    x1_original_N = normalizar(x1_original, x1_min, x1_max)
    x2_original_N = normalizar_ceros(x2_original, x2_min, x2_max)
    x1_original_memory = create_memory_matrix(x1_original_N, M)
    x2_original_memory = create_memory_matrix(x2_original_N, M)
    X_original = np.hstack((x1_original_memory, x2_original_memory))

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, X_original, x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max



# def generar_entradas_salidas(tipo_datos, M):

#     _, x1_train, x2_train, y1_train, y2_train = leer('dataset/train.txt', tipo_datos)
#     _, x1_valid, x2_valid, y1_valid, y2_valid = leer('dataset/valid.txt', tipo_datos)
#     _, x1_test, x2_test, y1_test, y2_test = leer('dataset/test.txt', tipo_datos)
#     time, x1_original, x2_original, y1_original, y2_original = leer('dataset/dataset.txt', tipo_datos)

#     # Identificar maximos y minimos del conjunto de entrenamiento
#     x1_min, x1_max = calcular_min_max(x1_train)
#     x2_min, x2_max = calcular_min_max(x2_train)
#     y1_min, y1_max = calcular_min_max(y1_train)
#     y2_min, y2_max = calcular_min_max(y2_train)

#     # Normalizar datos de entrenamiento
#     x1_train_N = normalizar(x1_train, x1_min, x1_max)
#     x2_train_N = normalizar(x2_train, x2_min, x2_max)
#     y1_train_N = normalizar(y1_train, y1_min, y1_max)
#     y2_train_N = normalizar(y2_train, y2_min, y2_max)

#     # Normalizar datos de validación
#     x1_valid_N = normalizar(x1_valid, x1_min, x1_max)
#     x2_valid_N = normalizar(x2_valid, x2_min, x2_max)
#     y1_valid_N = normalizar(y1_valid, y1_min, y1_max)
#     y2_valid_N = normalizar(y2_valid, y2_min, y2_max)

#     # Normalizar datos de prueba
#     x1_test_N = normalizar(x1_test, x1_min, x1_max)
#     x2_test_N = normalizar(x2_test, x2_min, x2_max)
#     y1_test_N = normalizar(y1_test, y1_min, y1_max)
#     y2_test_N = normalizar(y2_test, y2_min, y2_max)


#     # Crear las matrices de memoria para x1 y x2
#     x1_train_memory = create_memory_matrix(x1_train_N, M)
#     x2_train_memory = create_memory_matrix(x2_train_N, M)
#     x1_valid_memory = create_memory_matrix(x1_valid_N, M)
#     x2_valid_memory = create_memory_matrix(x2_valid_N, M)
#     x1_test_memory  = create_memory_matrix(x1_test_N, M)
#     x2_test_memory  = create_memory_matrix(x2_test_N, M)

#     # Combinar las entradas en una sola matriz de forma [L, 2 * M]
#     X_train = np.hstack((x1_train_memory, x2_train_memory))  # Dimensión (L, 2 * M)
#     X_valid = np.hstack((x1_valid_memory, x2_valid_memory))  # Dimensión (L, 2 * M)
#     X_test  = np.hstack((x1_test_memory, x2_test_memory))  # Dimensión (L, 2 * M)


#     y1_train_N = np.array(y1_train_N)
#     y2_train_N = np.array(y2_train_N)
#     y1_valid_N = np.array(y1_valid_N)
#     y2_valid_N = np.array(y2_valid_N)
#     y1_test_N = np.array(y1_test_N)
#     y2_test_N = np.array(y2_test_N)

#     # Combinar las salidas en una sola matriz de forma [L, 2]
#     Y_train = np.hstack((y1_train_N.reshape(-1, 1), y2_train_N.reshape(-1, 1)))  # Dimensión (L, 2)
#     Y_valid = np.hstack((y1_valid_N.reshape(-1, 1), y2_valid_N.reshape(-1, 1)))  # Dimensión (L, 2)
#     Y_test  = np.hstack((y1_test_N.reshape(-1, 1), y2_test_N.reshape(-1, 1)))  # Dimensión (L, 2)


#     x1_original_N = normalizar(x1_original, x1_min, x1_max)
#     x2_original_N = normalizar(x2_original, x2_min, x2_max)
#     x1_original_memory = create_memory_matrix(x1_original_N, M)
#     x2_original_memory = create_memory_matrix(x2_original_N, M)
#     X_original = np.hstack((x1_original_memory, x2_original_memory))  # Dimensión (L, 2 * M)

#     return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, X_original, x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max


# Función de pérdida NMSE en dB
def nmse_db_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    power_true = tf.reduce_mean(tf.square(y_true))
    nmse = mse / power_true
    nmse_db = 10 * tf.math.log(nmse) / tf.math.log(10.0)  # Convertir a dB
    return nmse_db

# Función para crear las matrices de memoria
def create_memory_matrix(data, M):
    L = len(data)
    X = np.zeros((L, M))
    for i in range(M):
        X[i:, i] = data[:L - i]
    return X


def generar_predicciones_ann(model, X_original, y1_min, y1_max, y2_min, y2_max, x2_original, x1_original):
    # Hacer las predicciones utilizando el modelo entrenado
    predictions = model.predict(X_original)

    # Mostrar las predicciones
    y1_pred, y2_pred = predictions[:, 0], predictions[:, 1]

    y1 = desnormalizar(y1_pred, y1_min, y1_max)
    y2 = desnormalizar(y2_pred, y2_min, y2_max)

    I_out = y1
    Q_out = y2

    # Si I_out y Q_out son listas, conviértelas a arrays de numpy
    I_out = np.array(I_out)
    Q_out = np.array(Q_out)

    # Cálculos predichos
    Vout_pred = np.sqrt(I_out**2 + Q_out**2)
    # fase_pred = np.arctan2(Q_out, I_out) - np.arctan2(x2_original, x1_original)
    # Calcular los ángulos utilizando np.arctan2
    angle_out = np.arctan2(Q_out, I_out)
    angle_original = np.arctan2(x2_original, x1_original)
    # Restringir los ángulos al rango [-pi/2, pi/2]
    angle_out_restricted = np.mod(angle_out + np.pi/2, np.pi) - np.pi/2
    angle_original_restricted = np.mod(angle_original + np.pi/2, np.pi) - np.pi/2
    # Calcular fase_pred como la resta de los ángulos restringidos
    fase_pred = angle_out_restricted - angle_original_restricted
    # Restringir fase_pred al rango [-pi/2, pi/2]
    fase_pred = np.mod(fase_pred + np.pi/2, np.pi) - np.pi/2

    return Vout_pred, fase_pred, I_out, Q_out