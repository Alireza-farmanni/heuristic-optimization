import numpy as np
from heuristic_opt.algorithms.swarm import PSO, GWO
from heuristic_opt.algorithms.evolutionary import CSO, MBO

# Define the continuous objective function
def convex_loss(x):
    return np.sum(x**2) - 12 * x[0] + 6

# Define shared optimization parameters
dim = 10
bounds = (-50.0, 50.0)
pop = 30
iters = 150

# Dictionary of instantiated algorithms
optimizers = {
    "PSO_Baseline": PSO(convex_loss, dim, bounds, pop, iters),
    "Grey_Wolf": GWO(convex_loss, dim, bounds, pop, iters),
    "Cuckoo_Search": CSO(convex_loss, dim, bounds, pop, iters),
    "Monarch_Butterfly": MBO(convex_loss, dim, bounds, pop, iters)
}

# Run and log each algorithm's performance
for name, optimizer in optimizers.items():
        
    # Run optimization
    best_pos, best_score = optimizer.optimize()      
    print(f"{name} finished with loss: {best_score:.4f}")