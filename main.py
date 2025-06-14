from classic_de import ClassicDE
from de import DiversityDE
import numpy as np

def sphere(x):
    return np.sum(x**2)

# Parametry DE
pop_size = 20
dim = 5
bounds = [(-5, 5)] * dim
F = 0.5
CR = 0.9

# Uruchomienie ClassicDE
classic_de = ClassicDE(pop_size, dim, bounds, F, CR)
best_sol_c, best_val_c, history_c = classic_de.run(sphere, max_evals=1000)

# Uruchomienie DiversityDE
div_de = DiversityDE(pop_size, dim, bounds, F, CR, delta=1e-2, k=2, replacement_mode='mixed')
best_sol_d, best_val_d, history_d = div_de.run(sphere, max_evals=1000)

print("ClassicDE Best Value:", best_val_c)
print("DiversityDE Best Value:", best_val_d)