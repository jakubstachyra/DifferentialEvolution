import classic_de
import numpy as np

def diversity_metric(population):
    """
    Metryka różnorodności: odchylenie standardowe odległości euklidesowych
    pomiędzy wszystkimi parami osobników w populacji.
    """
    pop = population
    n = pop.shape[0]
    # Oblicz wszystkie pary odległości
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(pop[i] - pop[j]))
    dists = np.array(dists)
    return np.std(dists)


class DiversityDE(classic_de.ClassicDE):
    """
    Rozszerzenie klasycznego DE o mechanizm zwiększania różnorodności.
    """

    def __init__(self, pop_size, dim, bounds, F=0.5, CR=0.9,
                 delta=1e-3, k=2, replacement_mode='random'):
        """
        delta: próg różnorodności
        k: liczba najsłabszych osobników do zastąpienia
        replacement_mode: 'random', 'perturb', 'crossover', albo 'mixed'
        """
        super().__init__(pop_size, dim, bounds, F, CR)
        self.delta = delta
        self.k = k
        self.replacement_mode = replacement_mode

    def generate_random(self):
        """
        Generuje losowy osobnik wewnątrz granic.
        """
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def perturb_best(self):
        """
        Generuje nowy osobnik przez perturbację jednego z najlepszych.
        """

        best_idx = np.argmin(self.fitness)
        best = self.population[best_idx]

        noise = np.random.normal(0, 1, size=self.dim)
        candidate = best + 1.5 * noise  # 1.5 to przykładowa skala perturbacji

        return np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])

    def crossover_random(self):
        """
        Generuje nowy osobnik przez krzyżowanie dwóch losowych.
        """

        idxs = np.random.choice(self.pop_size, 2, replace=False)
        parent1 = self.population[idxs[0]]
        parent2 = self.population[idxs[1]]

        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        return np.clip(child, self.bounds[:, 0], self.bounds[:, 1])

    def replace_weakest(self):
        """
        Zastępuje k najsłabszych osobników nowo wygenerowanymi zgodnie
        z wybraną strategią replacement_mode.
        """

        sorted_idxs = np.argsort(self.fitness)
        weakest = sorted_idxs[-self.k:]  # indeksy najsłabszych

        for idx in weakest:
            if self.replacement_mode == 'random':
                new_ind = self.generate_random()
            elif self.replacement_mode == 'perturb':
                new_ind = self.perturb_best()
            elif self.replacement_mode == 'crossover':
                new_ind = self.crossover_random()
            elif self.replacement_mode == 'mixed':

                choice = np.random.choice(['random', 'perturb', 'crossover'])
                if choice == 'random':
                    new_ind = self.generate_random()
                elif choice == 'perturb':
                    new_ind = self.perturb_best()
                else:
                    new_ind = self.crossover_random()
            else:
                raise ValueError("Niepoprawna strategia zastępowania.")

 
            self.population[idx] = new_ind
            self.fitness[idx] = func(new_ind)  # zakładamy, że func jest dostępna
            self.eval_count += 1

    def run(self, func, max_evals):
        """
        Uruchomienie zmodyfikowanego DE z mechanizmem różnorodności.
        """
        fitness = self.evaluate(func)
        best_idx = np.argmin(fitness)
        best_solution = self.population[best_idx].copy()
        best_value = fitness[best_idx]

        history = {
            'evals': [self.eval_count],
            'best_values': [best_value],
            'diversity': [diversity_metric(self.population)]
        }


        while self.eval_count < max_evals:
            new_population = np.zeros_like(self.population)
            new_fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):

                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)


                trial_fit = func(trial)
                self.eval_count += 1

                if trial_fit <= self.fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fit
                else:
                    new_population[i] = self.population[i]
                    new_fitness[i] = self.fitness[i]


                if new_fitness[i] < best_value:
                    best_value = new_fitness[i]
                    best_solution = new_population[i].copy()

               
                if self.eval_count >= max_evals:
                    break


            self.population = new_population
            self.fitness = new_fitness


            current_div = diversity_metric(self.population)
            if current_div < self.delta:

                self.replace_weakest()


                current_div = diversity_metric(self.population)

            # Zapis historii
            history['evals'].append(self.eval_count)
            history['best_values'].append(best_value)
            history['diversity'].append(current_div)

        return best_solution, best_value, history

