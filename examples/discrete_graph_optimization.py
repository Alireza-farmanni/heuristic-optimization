from heuristic_opt.algorithms.ant_colony import AntColonyOptimizer

# Define node distances (0 indicates no direct path or self-loop)
adjacency_matrix = [
    [0, 10, 4, 2, 6],
    [10, 0, 2, 20, 25],
    [4, 2, 0, 6, 9],
    [2, 20, 6, 0, 1],
    [10, 5, 6, 7, 0]
]

aco = AntColonyOptimizer(
    cost_matrix=adjacency_matrix,
    pop_size=50,
    max_iter=100,
    ro=0.05  # Changed from evaporation_rate to ro
)

best_node, lowest_cost = aco.optimize()

print("Optimization Complete.")
print(f"Final Best Cost: {lowest_cost}")
print(f"Target Node Reached: {best_node}")