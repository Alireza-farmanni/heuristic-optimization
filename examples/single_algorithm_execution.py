import numpy as np
from heuristic_opt.algorithms.swarm import PSO

# 1. Define a continuous objective function (e.g., optimizing 5 parameters)
def convex_loss(x):
    return np.sum(x**2) - 12 * x[0] + 6

# 2. Instantiate the optimizer
pso = PSO(
    obj_func=convex_loss,
    dim=5,
    bounds=(-10.0, 10.0),
    pop_size=40,
    max_iter=200,
    c1=2.0, 
    c2=2.0
)

# 3. Execute the loop
best_params, best_loss = pso.optimize()

print(f"Optimal Parameters: {best_params}")
print(f"Minimum Loss: {best_loss}")