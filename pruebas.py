from Genetico import Hiperparametros, algoritmo_genetico
from time import time
from csv import writer

with open("Results/partidas.csv") as file:
    output = writer(file)
    output.writerow(["partidas", "fitness", "tiempo"])
    for partidas in [1, 5, 10, 20]:
        hp = Hiperparametros(partidas=partidas)
        t0 = time()
        fitness, individuo = algoritmo_genetico(epochs=20, HIPERPARAMETROS=hp)
        tiempo = time() - t0
        output.writerow(partidas, fitness, tiempo)

with open("Results/tamaño.csv") as file:
    output = writer(file)
    output.writerow(["tamaño", "fitness", "tiempo"])
    for tamaño in [1, 5, 10, 20, 50]:
        hp = Hiperparametros(tamaño=tamaño)
        t0 = time()
        fitness, individuo = algoritmo_genetico(epochs=20, HIPERPARAMETROS=hp)
        tiempo = time() - t0
        output.writerow(tamaño, fitness, tiempo)

with open("Results/mutate_prob.csv") as file:
    output = writer(file)
    output.writerow(["mutate_prob", "fitness", "tiempo"])
    for mutate_prob in [1, 5, 10, 20, 50]:
        hp = Hiperparametros(mutate_prob=mutate_prob)
        t0 = time()
        fitness, individuo = algoritmo_genetico(epochs=20, HIPERPARAMETROS=hp)
        tiempo = time() - t0
        output.writerow(mutate_prob, fitness, tiempo)
