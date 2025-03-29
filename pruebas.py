from Genetico import Hiperparametros, algoritmo_genetico
from time import time
from csv import writer

with open("Results/partidas.csv", "w") as file:
    output = writer(file)
    output.writerow(["partidas", "fitness", "tiempo"])
    for partidas in [10, 20, 30, 40]:
        hp = Hiperparametros(partidas=partidas)
        t0 = time()
        fitness, individuo = algoritmo_genetico(epochs=20, HIPERPARAMETROS=hp)
        tiempo = time() - t0
        output.writerow([partidas, fitness, tiempo])

with open("Results/tamaño.csv", "w") as file:
    output = writer(file)
    output.writerow(["tamaño", "fitness", "tiempo"])
    for tamaño in [50, 100, 150]:
        hp = Hiperparametros(tamaño=tamaño)
        t0 = time()
        fitness, individuo = algoritmo_genetico(epochs=20, HIPERPARAMETROS=hp)
        tiempo = time() - t0
        output.writerow([tamaño, fitness, tiempo])

with open("Results/mutate_prob.csv", "w") as file:
    output = writer(file)
    output.writerow(["mutate_prob", "fitness", "tiempo"])
    for mutate_prob in [0.1, 0.2, 0.3, 0.4]:
        hp = Hiperparametros(mutate_prob=mutate_prob)
        t0 = time()
        fitness, individuo = algoritmo_genetico(epochs=20, HIPERPARAMETROS=hp)
        tiempo = time() - t0
        output.writerow([mutate_prob, fitness, tiempo])
