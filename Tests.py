from Genetico import algoritmo_genetico, Hiperparametros
from csv import writer


if __name__ == "__main__":

    def run_tests(name: str, tests: list[Hiperparametros]):
        with open(f"Results/{name}.txt", "w") as file:
            log = writer(file, delimiter="\t")
            log.writerow([name, "mean"])

            for hp in tests:
                results = []
                for _ in range(5):
                    fitness, _ = algoritmo_genetico(10, hp)
                    results.append(fitness)
                log.writerow([p_mut, sum(results) / len(results)])

    tests = []
    for p_mut in [0.01, 0.05, 0.1, 0.2]:
        tests.append(Hiperparametros(mutate_prob=p_mut))

    run_tests("mutate_prob", tests)
