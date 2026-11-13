import numpy as np

# Cargar datos desde el archivo, especificando el delimitador de tabulación
data = np.loadtxt('dataset/dataset.txt', delimiter='\t')

# Mezclar los datos de forma aleatoria
# np.random.shuffle(data)

# Obtener el número total de filas
total_rows = data.shape[0]

# Calcular el número de filas para cada conjunto
train_size = int(0.7 * total_rows)
valid_size = int(0.15 * total_rows)
test_size = total_rows - train_size - valid_size

# Dividir los datos en conjuntos
train_data = data[:train_size]
valid_data = data[train_size:train_size + valid_size]
test_data = data[train_size + valid_size:]

# Guardar los conjuntos en archivos .txt con tabulaciones
np.savetxt('dataset/train.txt', train_data, delimiter='\t', fmt='%.9f')
np.savetxt('dataset/valid.txt', valid_data, delimiter='\t', fmt='%.9f')
np.savetxt('dataset/test.txt', test_data, delimiter='\t', fmt='%.9f')

print(f'División completada: {train_size} para entrenamiento, {valid_size} para validación, {test_size} para prueba.')
