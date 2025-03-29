from dataclasses import dataclass
import random
import numpy as np
from multiprocessing import Pool
from itertools import islice
from csv import writer
from time import time
from argparse import ArgumentParser

from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta

from Managers.GameDirector import GameDirector

AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]


@dataclass
class Hiperparametros:
    partidas: int = 10
    tamaño: int = 100
    mutate_prob: float = 0.2
    mutate_var: float = 0.2
    seleccion_m: int = 30
    seleccion_n: int = 30


def simular(individuo):
    chosen_agent = random.choices(AGENTS, individuo)[0]
    all_agents = random.sample(AGENTS, 3) + [chosen_agent]
    random.shuffle(all_agents)

    try:
        game_director = GameDirector(
            agents=all_agents, max_rounds=200, store_trace=False
        )
        game_trace = game_director.game_start(print_outcome=False)
    except Exception as e:
        print(f"Error: {e}")
        return 0

    last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
    last_turn = max(
        game_trace["game"][last_round].keys(),
        key=lambda t: int(t.split("_")[-1].lstrip("P")),
    )
    victory_points = game_trace["game"][last_round][last_turn]["end_turn"][
        "victory_points"
    ]

    chosen_player = f"J{all_agents.index(chosen_agent)}"
    sorted_agents = sorted(victory_points.items(), key=lambda x: int(x[1]))

    places = {a: i for i, (a, _) in enumerate(sorted_agents)}
    return places[chosen_player] + 1


def algoritmo_genetico(epochs, HIPERPARAMETROS=None):
    if HIPERPARAMETROS is None:
        HIPERPARAMETROS = Hiperparametros()

    pool = Pool(5)

    def softmax(individuo):
        e = np.exp(individuo)
        return e / e.sum()

    def crear_individuo():
        return softmax(np.random.rand(len(AGENTS)))

    def crear_poblacion():
        poblacion = []
        for _ in range(HIPERPARAMETROS.tamaño):
            individuo = crear_individuo()
            poblacion.append((fitness(individuo), individuo))
        return poblacion

    def fitness(individuo):
        positions = pool.map(simular, [individuo] * HIPERPARAMETROS.partidas)
        return sum(positions) / len(positions)

    def mutate(individuo):
        return individuo + HIPERPARAMETROS.mutate_var * (
            np.random.rand(len(AGENTS)) < HIPERPARAMETROS.mutate_prob
        )

    def crossover(individuo1, individuo2):
        points = [
            random.randint(0, len(AGENTS) - 2),
            random.randint(0, len(AGENTS) - 2),
        ]
        points = sorted(points)
        p1, p2 = points
        p2 += 1

        hijo1 = individuo1.copy()
        hijo2 = individuo2.copy()

        hijo1[p1:p2] = individuo2[p1:p2]
        hijo2[p1:p2] = individuo1[p1:p2]

        return hijo1, hijo2

    def seleccion(poblacion):
        muestra = random.sample(poblacion, HIPERPARAMETROS.seleccion_m)
        muestra = sorted(muestra, key=lambda x: x[0])
        muestra = list(map(lambda x: x[1], muestra))
        return muestra[: HIPERPARAMETROS.seleccion_n]

    def criba(poblacion):
        return sorted(poblacion, key=lambda x: x[0])[: HIPERPARAMETROS.tamaño]

    # Disponible en itertools en Python 3.12
    def batched(iterable, n, *, strict=False):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        batch = islice(iterator, n)
        while batch:
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch

    with open("log.txt", "w") as file:
        log = writer(file, delimiter="\t")
        log.writerow(["Epoch", "Best", "Mean"])

        poblacion = crear_poblacion()
        print("inicio")

        for i in range(epochs):
            positions = list(map(lambda x: x[0], poblacion))
            log.writerow([i, min(positions), sum(positions) / len(positions)])

            print(i)
            padres = seleccion(poblacion)
            random.shuffle(padres)

            hijos = []
            for p1, p2 in batched(padres, n=2):
                hijo1, hijo2 = crossover(p1, p2)

                hijo1 = mutate(hijo1)
                hijos.append((fitness(hijo1), hijo1))

                hijo2 = mutate(hijo2)
                hijos.append((fitness(hijo2), hijo2))

            poblacion.extend(hijos)
            poblacion = criba(poblacion)

    pool.close()
    mejor_fitness, mejor_individuo = min(poblacion, key=lambda x: x[0])
    return mejor_fitness, softmax(mejor_individuo)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--partidas", type=int, default=10)
    parser.add_argument("--tamaño", type=int, default=100)
    parser.add_argument("--mutate_prob", type=float, default=0.2)
    parser.add_argument("--mutate_var", type=float, default=0.2)
    parser.add_argument("--seleccion_m", type=int, default=30)
    parser.add_argument("--seleccion_n", type=int, default=30)

    args, _ = parser.parse_known_args()

    t0 = time()
    print(
        algoritmo_genetico(
            20, HIPERPARAMETROS=Hiperparametros(**dict(args._get_kwargs()))
        )
    )
    print(f"{(time() - t0) / 60:.2f} minutos")
    pool.close()
