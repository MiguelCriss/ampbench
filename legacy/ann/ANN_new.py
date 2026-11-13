import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from funciones_ANN import *

import sys
import os
# Agregar el directorio padre a sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Importar funciones desde el script en volterra
from metodos_tradicionales.funciones import *

# Tipo de datos (tipo=0 : vin,vout) (tipo=1 : I,Q)
tipo_datos = 1

# Parámetros
M = 7  # Profundidad de memoria

(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, X_original,
 x1_min, x1_max, x2_min, x2_max, y1_min, y1_max, y2_min, y2_max) = generar_entradas_salidas(tipo_datos, M)

time, x1_original, x2_original, y1_original, y2_original = leer('dataset/dataset.txt', tipo_datos)


# Definir el modelo de la red neuronal
model = Sequential()
model.add(Dense(64, input_shape=(2 * M,), activation='relu'))  # Capa oculta con 64 neuronas
model.add(Dense(32, activation='relu'))                       # Capa oculta con 32 neuronas
model.add(Dense(2, activation='linear'))                      # Capa de salida con 2 neuronas (para y1 y y2)



from keras.optimizers import Adam 
optimizer = Adam(learning_rate=0.0001)  # Tasa de aprendizaje inicial más baja

loss = nmse_db_loss
# Compilar el modelo
model.compile(optimizer, loss, metrics=['mae'])



# Entrenar el modelo con datos de validación personalizados
history = model.fit(X_train, Y_train, 
                    epochs=300, 
                    batch_size=320, 
                    validation_data=(X_valid, Y_valid))

# Graficar NMSE en dB durante el entrenamiento y la validación
plt.plot(history.history['loss'], label='NMSE dB - Entrenamiento')
plt.plot(history.history['val_loss'], label='NMSE dB - Validación')
plt.xlabel('Época')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.title('NMSE (dB) durante Entrenamiento y Validación')
plt.show()




# Evaluar el modelo
# loss, mae = model.evaluate(X_test, Y_test)
# print(f"Loss: {loss}, Mean Absolute Error: {mae}")

_, x1_test, x2_test, y1_test, y2_test = leer('dataset/test.txt', tipo_datos)
################# predicciones test
predictions = model.predict(X_test)

# Mostrar las predicciones
y1_pred, y2_pred = predictions[:, 0], predictions[:, 1]

y1 = desnormalizar(y1_pred, y1_min, y1_max)
y2 = desnormalizar(y2_pred, y2_min, y2_max)

I_pred_test = y1
Q_pred_test = y2

# Si I_out y Q_out son listas, conviértelas a arrays de numpy
I_pred_test = np.array(I_pred_test)
Q_pred_test = np.array(Q_pred_test)

################# predicciones
Vout_pred, fase_pred, I_out_pred, Q_out_pred = generar_predicciones_ann(model, X_original, y1_min, y1_max, 
                                                              y2_min, y2_max, x2_original, x1_original)


error_metric = 'NMSE_dB'

# Cálculos originales
Vin_original = np.sqrt(x1_original**2 + x2_original**2)
Vout_original = np.sqrt(y1_original**2 + y2_original**2)
# fase_original = np.arctan2(y2_original, y1_original) - np.arctan2(x2_original, x1_original)
# Calcular los ángulos utilizando np.arctan2
angle_y_original = np.arctan2(y2_original, y1_original)
angle_x_original = np.arctan2(x2_original, x1_original)

# Restringir los ángulos al rango [-pi/2, pi/2]
angle_y_original_restricted = np.mod(angle_y_original + np.pi/2, np.pi) - np.pi/2
angle_x_original_restricted = np.mod(angle_x_original + np.pi/2, np.pi) - np.pi/2

# Calcular fase_original como la resta de los ángulos restringidos
fase_original = angle_y_original_restricted - angle_x_original_restricted

# Restringir fase_original al rango [-pi/2, pi/2]
fase_original = np.mod(fase_original + np.pi/2, np.pi) - np.pi/2


I_out_orig = y1_original
Q_out_orig = y2_original
print("-------------------------------")
print(f"{error_metric} test = {calculate_error(y1_test+1j*y2_test, I_pred_test+1j*Q_pred_test, error_metric)}")
print("-------------------------------")
print(f"{error_metric} = {calculate_error(I_out_orig+1j*Q_out_orig, I_out_pred+1j*Q_out_pred, error_metric)}")

print("-------------------------------")
print(f"{error_metric} Vout = {calculate_error(Vout_original, Vout_pred, error_metric)}")
print(f"{error_metric} Fase = {calculate_error(fase_original, fase_pred, error_metric)}")
print("-------------------------------")


modelo = 'ANN'


# GRAFICAS
N=1

import pickle
# Guardar los datos en un archivo
with open('datos_salida.pkl', 'wb') as file:
    pickle.dump({'vin': Vin_original, 'vout': Vout_pred, 'fase': fase_pred}, file)
print('OK')

graficas(time, Vin_original, Vout_original, Vout_pred, fase_original, fase_pred, modelo, N, M,
          I_out_pred, Q_out_pred, I_out_orig, Q_out_orig)



import pickle
# Guardar los datos en un archivo
with open('timeseries.pkl', 'wb') as file:
        pickle.dump({'t': time, 'vout': Vout_pred, 'vout_orig': Vout_original}, file)

print("Datos guardados exitosamente en 'timeseries.pkl'")


