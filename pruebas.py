from Genetico import Hiperparametros, algoritmo_genetico
from time import time
from csv import writer

# 8 minutos por ejecución
with open("resultados/partidas.csv") as file:
    output = writer(file)
    output.writerow(["partidas", "fitness", "tiempo"])
    for partidas in [1, 5, 10, 20, 50]:
        hp = Hiperparametros(partidas=partidas)
        t0 = time()
        fitness, individuo = algoritmo_genetico(epochs=20, HIPERPARAMETROS=hp)
        tiempo = time() - t0
        output.writerow(partidas, fitness, tiempo)
