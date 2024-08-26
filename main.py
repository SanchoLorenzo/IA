import winsound
import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz

# Hiperparámetros
train_percentage = 0.70
n = 2
units = 8
activation = "sigmoid"
learning_rate = 0.005
loss = "categorical_crossentropy"
batch_size = 10
epochs = 125

# Lectura de los datos
dataset = pandas.read_excel("Mine_Dataset.xlsx")
print(dataset)

for i in range(1, 6):  # Para valores de 1 a 5
    dataset[f'M{i}'] = (dataset["M"] == i).astype(int)

# Eliminar la columna original si ya no es necesaria
dataset.drop(columns=["M"], inplace=True)


# División en entrenamiento y validación.
trainset = dataset.sample(frac=train_percentage)  # ATENCIÓN: HIPERPARÁMETRO
# Se extraen datos para el entrenamiento

testset = dataset.drop(trainset.index)  # Y se le quitan esos mismos
# al dataset para crear los datos de prueba
print(trainset)


# Inicialización del modelo
network = models.Sequential()

# Declaración de la capa de entrada
network.add(Input(shape=(3,)))  #entradas

# Ciclo de capas de neuronas intermedias
for i in range(n-1):  # n: número de neuronas intermedias
    network.add(layers.Dense(
            units=units,  # units: número de neuronas por capa
            activation=activation))  # activation: función de activación elegida.
# VER DOCUMENTACIÓN PARA VER LAS POSIBLES OPCIONES A ELEGIR

# Declaración de la capa de salida
network.add(layers.Dense(
        units=5,  # Una única salida
        activation="sigmoid"))  # Problema de regresión, por tanto salida dada por sigmoide

network.compile(
        optimizer=tf.keras.optimizers.Adam(  # optimizer: algoritmo de optimización
            learning_rate=learning_rate  # learning_rate: ritmo de aprendizaje
        ),
        loss=loss)  # loss: función de pérdida

losses = network.fit(x=trainset[['V', 'H', 'S']],  # Datos de entrada (entrenamiento)
                     y=trainset[['M1', 'M2', 'M3', 'M4', 'M5']],  # Datos de salida (entrenamiento)
                     validation_data=( # Conjuntos de datos de validación
                            testset[['V', 'H', 'S']], # Entrada de validación
                            testset[['M1', 'M2', 'M3', 'M4', 'M5']]  # Salida de validación
                            ),
                     batch_size=batch_size,  # Tamaño de muestreo
                     epochs=epochs  # Cantidad de iteraciones de entrenamiento
                     )

winsound.Beep(350, 500)  # Aviso auditivo de la finalización del entrenamiento.

# Se extrae el historial de error contra iteraciones de la clase
loss_df = pandas.DataFrame(losses.history)

loss_df.loc[:, ['loss', 'val_loss']].plot() # Se crea la curva a graficar

# Y se llama a la ventana que se muestre la grafica
plt.show()

# Se eligen 10 datos al azar
dato = dataset.sample(frac=10/dataset.shape[0])

# Se predice el precio que tendría, según el modelo
datoPrueba = dato.drop(columns=[['Null', 'Anti-Tank', 'Anti-personnel', 'Booby Trapped', 'M14']])

# Se revierte la normalización al mostrar los resultados
#  print("Dato ingresado:")
#  print(dato*difference+ min_val)
#  dato["Price"] = precio
#  print("Estimacion: ")
#  print(dato*difference+ min_val)

