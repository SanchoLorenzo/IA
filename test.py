import pandas
# Lectura de los datos
dataset = pandas.read_excel("Mine_Dataset.xlsx")
print(dataset)

for i in range(1, 6):  # Para valores de 1 a 5
    dataset[f'M_{i}'] = (dataset["M"] == i).astype(int)

# Eliminar la columna original si ya no es necesaria
dataset.drop(columns=["M"], inplace=True)

print(dataset)