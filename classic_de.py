import numpy as np

class ClassicDE:
    """
    Klasa implementująca klasyczną ewolucję różnicową (DE/rand/1/bin).
    """

    def __init__(self, pop_size, dim, bounds, F=0.5, CR=0.9):
        """
        Inicjalizacja parametrów DE.
        
        pop_size: Rozmiar populacji (N)
        dim: Wymiar problemu
        bounds: Lista krotek (min, max) dla każdego wymiaru
        F: Współczynnik mutacji
        CR: Współczynnik krzyżowania
        """
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = np.array(bounds)
        self.F = F
        self.CR = CR

        # Inicjalizacja populacji losowo wewnątrz granic
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(pop_size, dim)
        )
        self.fitness = None  # będzie wypełniane przy ocenie
        self.eval_count = 0

    def evaluate(self, func):
        """
        Ocena wszystkich osobników w populacji za pomocą funkcji celu.
        Zwraca wektor wartości funkcji celu (im mniejsze, tym lepsze).
        """
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.eval_count += self.pop_size
        return self.fitness

    def mutate(self, idx):
        """
        Mutacja DE/rand/1: wybór r1, r2, r3 różne od idx i r1 != r2 != r3.
        """
        indices = list(range(self.pop_size))
        indices.remove(idx)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        x_r1 = self.population[r1]
        x_r2 = self.population[r2]
        x_r3 = self.population[r3]
        mutant = x_r1 + self.F * (x_r2 - x_r3)

        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
        return mutant

    def crossover(self, target, mutant):
        """
        Crossover binarny: tworzy wektor próbny u.
        target: wektor x_i
        mutant: wektor v_i
        """
        trial = np.copy(target)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def select(self, idx, trial, trial_fit):
        """
        Selekcja DE: jeśli fit(trial) <= fit(target), to trial zastępuje target.
        Zwraca (wektor_do_następnej_generacji, jego_fitness).
        """
        if trial_fit <= self.fitness[idx]:
            return trial, trial_fit
        else:
            return self.population[idx], self.fitness[idx]

    def run(self, func, max_evals):
        """
        Uruchomienie klasycznego algorytmu DE.
        
        func: funkcja celu przyjmująca wektor wymiaru dim.
        max_evals: maksymalna liczba wywołań funkcji celu.
        
        Zwraca:
        - best_solution: najlepszy znaleziony wektor
        - best_value: wartość funkcji celu dla best_solution
        - history: słownik z przebiegiem najlepszej wartości i ewaluacji
        """

        fitness = self.evaluate(func)
        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx].copy()
        best_value = fitness[best_idx]

        history = {
            'evals': [self.eval_count],
            'best_values': [best_value]
        }


        while self.eval_count < max_evals:
            new_population = np.zeros_like(self.population)
            new_fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                # Mutacja
                mutant = self.mutate(i)
                # Crossover
                trial = self.crossover(self.population[i], mutant)
                # Ocena próbnego
                trial_fit = func(trial)
                self.eval_count += 1

                # Selekcja
                chosen_vec, chosen_fit = self.select(i, trial, trial_fit)
                new_population[i] = chosen_vec
                new_fitness[i] = chosen_fit

                # Aktualizacja najlepszego
                if chosen_fit < best_value:
                    best_value = chosen_fit
                    best_solution = chosen_vec.copy()

                # Przerwij jeśli przekroczono limit
                if self.eval_count >= max_evals:
                    break

            # Aktualizacja populacji i fitness
            self.population = new_population
            self.fitness = new_fitness

            # Zapis historyczny
            history['evals'].append(self.eval_count)
            history['best_values'].append(best_value)

        return best_solution, best_value, history
