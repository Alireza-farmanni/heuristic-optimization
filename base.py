import numpy as np
from abc import ABC, abstractmethod

import numpy as np
from abc import ABC, abstractmethod

class BaseContinuousOptimizer(ABC):
    def __init__(self, objective_function, dim, bounds, pop_size, max_iter):
        self.objective_function = objective_function
        self.dim = dim
        self.lb, self.ub = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # Initialize population
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.costs = np.apply_along_axis(self.objective_function, 1, self.positions)
        
        self.global_best_position = self.positions[np.argmin(self.costs)].copy()
        self.global_best_score = np.min(self.costs)
        self.history = []

    def enforce_bounds(self):
        self.positions = np.clip(self.positions, self.lb, self.ub)

    def update_costs_and_bests(self, new_costs):
        """Updates internal costs and tracks the global best."""
        self.costs = new_costs
        current_best_idx = np.argmin(self.costs)
        if self.costs[current_best_idx] < self.global_best_score:
            self.global_best_score = self.costs[current_best_idx]
            self.global_best_position = self.positions[current_best_idx].copy()

    @abstractmethod
    def step(self, t):
        """Defines the position update logic for a single iteration."""
        pass

    def optimize(self):
        """The standard optimization loop."""
        for t in range(self.max_iter):
            self.step(t)
            self.enforce_bounds()
            new_costs = np.apply_along_axis(self.objective_function, 1, self.positions)
            self.update_costs_and_bests(new_costs)
            self.history.append(self.global_best_score)
        return self.global_best_position, self.global_best_score


class BaseDiscreteOptimizer(ABC):
    """Base class for graph-based discrete algorithms like ACO."""
    def __init__(self, cost_matrix, pop_size, max_iter):
        self.cost_matrix = np.array(cost_matrix, dtype=float)
        self.num_nodes = len(self.cost_matrix)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.history = []

    @abstractmethod
    def step(self):
        pass

    def optimize(self):
        for _ in range(self.max_iter):
            self.step()
            self.history.append(self.global_best_score)
        return self.global_best_position, self.global_best_score