import winsound
import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz

# Hiperparámetros
train_percentage = 0.75
n = 2
units = 5
activation = "sigmoid"
learning_rate = 0.01
loss = "mse"
batch_size = 250
epochs = 25



# Lectura de los datos
dataset = pandas.read_excel("Data_set.xlsx")
print(dataset)

# Truncado de datos
dataset = dataset.drop(columns="Airline")  # Se elimina la columna "Airline"
print(dataset)

for category in dataset:  # Se procede a eliminar toda información que bo esté  
    if category != "Duration":  # relacionada con las elegidas para el análisis.
        if category != "Price":
            dataset = dataset.drop(columns=category)
print(dataset)

# Función que convierte los datos no numéricos de
# la propiedad "Duration" en numéricos.
# Entrada: stringDtype; Salida: integer
def uniform_duration(data):
    if data.find("m") == -1:  # Dato con formato #h
        [horas] = data.split(" ")  # Asigna a "horas"
        horas = horas.strip("h")  # el número de horas
        minutos = 0  # y a "minutos" 0
    elif data.find("h") == -1:  # Dato con formato #m
        [minutos] = data.split(" ")  # Asigna a "minutos"
        minutos = minutos.strip("m")  # el número de minutos
        horas = 0  # y a "horas" 0
    else:
        [horas, minutos] = data.split(" ")  # Dato con formato #h #m
        minutos = minutos.strip("m")  # Extrae las horas y minutos
        horas = horas.strip("h")  # y los asigna a la variable apropiada
    # Se suma la cantidad de minutos totales que durará el vuelo
    data = int(horas)*60 + int(minutos)
    return data


# Procesamiento de datos, por medio del llamado a la función
dataset["Duration"] = dataset["Duration"].apply(lambda x: uniform_duration(x))
print(dataset)

# Normalizado
max_val = dataset.max(axis=0)  # Se obtiene el máximo de cada columna
min_val = dataset.min(axis=0)  # Se obtiene el mínimo de cada columna
difference = max_val - min_val  # Se obtiene la diferencia de los dos
dataset = (dataset - min_val)/(difference)  # Y se utiliza para normalizarlas
dataset = dataset.astype(float)  # Se asegura que los datos sean tipo float

# División en entrenamiento y validación.
trainset = dataset.sample(frac=train_percentage)  # ATENCIÓN: HIPERPARÁMETRO
# Se extraen datos para el entrenamiento

testset = dataset.drop(trainset.index)  # Y se le quitan esos mismos
# al dataset para crear los datos de prueba
print(trainset)

# Inicialización del modelo
network = models.Sequential()

# Declaración de la capa de entrada
network.add(Input(shape=(1,)))

# Ciclo de capas de neuronas intermedias
for i in range(n-1):  # n: número de neuronas intermedias
    network.add(layers.Dense(
            units=units,  # units: número de neuronas por capa
            activation=activation))  # activation: función de activación elegida.
            # VER DOCUMENTACIÓN PARA VER LAS POSIBLES OPCIONES A ELEGIR

# Declaración de la capa de salida
network.add(layers.Dense(
        units=1,  # Una única salida
        activation="sigmoid"))  # Problema de regresión, por tanto salida dada por sigmoide

network.compile(
        optimizer=tf.keras.optimizers.Adam(  # optimizer: algoritmo de optimización
            learning_rate=learning_rate  # learning_rate: ritmo de aprendizaje
        ),
        loss=loss)  # loss: función de pérdida

losses = network.fit(x=trainset["Duration"],  # Datos de entrada (entrenamiento)
                     y=trainset['Price'],  # Datos de salida (entrenamiento)
                     validation_data=( # Conjuntos de datos de validación
                            testset["Duration" ], # Entrada de validación
                            testset['Price']  # Salida de validación
                            ),
                     batch_size=150,  # Tamaño de muestreo
                     epochs=50  # Cantidad de iteraciones de entrenamiento
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
datoPrueba = dato.drop(columns=["Price"])

# Se revierte la normalización al mostrar los resultados
precio = network.predict(datoPrueba)
print("Dato ingresado:")
print(dato*difference+ min_val)
dato["Price"] = precio
print("Estimacion: ")
print(dato*difference+ min_val)
