import numpy as np
import random
from heuristic_opt.base import BaseDiscreteOptimizer

class AntColonyOptimizer(BaseDiscreteOptimizer):
    def __init__(self, cost_matrix, pop_size=100, max_iter=100, 
                 Q=10, alpha=1.0, beta=1.0, ro=0.01):
        super().__init__(cost_matrix, pop_size, max_iter)
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.ro = ro
        
        # Initialize pheromone matrix with 1s, leaving 0s on the diagonal 
        self.pheromones = np.ones((self.num_nodes, self.num_nodes))
        np.fill_diagonal(self.pheromones, 0)
        
        # Tracking state
        self.positions = np.zeros(self.pop_size, dtype=int)
        self.ant_costs = np.zeros(self.pop_size)
        
        # Random initial placement for the first step
        for i in range(self.pop_size):
            self.positions[i] = random.randint(0, self.num_nodes - 1)

    def step(self):
        best_next_nodes = np.zeros(self.pop_size, dtype=int)
        delta_tav = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.pop_size):
            current_node = self.positions[i]
            transition_prob = np.zeros(self.num_nodes)
            sum_of_p = 0.0
            
            # Calculate denominator for transition probabilities
            for j in range(self.num_nodes):
                if j != current_node and self.cost_matrix[current_node, j] != 0:
                    sum_of_p += (self.pheromones[current_node, j] ** self.alpha) * \
                                (1.0 / (self.cost_matrix[current_node, j] ** self.beta))
            
            # Calculate probabilities
            for j in range(self.num_nodes):
                if j != current_node and self.cost_matrix[current_node, j] != 0:
                    transition_prob[j] = ((self.pheromones[current_node, j] ** self.alpha) * \
                                         (1.0 / (self.cost_matrix[current_node, j] ** self.beta))) / sum_of_p
            
            # Select next node (using argmax as in the original implementation)
            best_next_nodes[i] = np.argmax(transition_prob)
            
            # Update costs
            self.ant_costs[i] += self.cost_matrix[current_node, best_next_nodes[i]]
            
            # Calculate local delta_tav contribution
            for j in range(self.num_nodes):
                if j == best_next_nodes[i] and j != current_node:
                    delta_tav[current_node, j] += self.Q / self.cost_matrix[current_node, j]

        # Evaporate and update global pheromones
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromones[i, j] = (1 - self.ro) * self.pheromones[i, j] + delta_tav[i, j]
                
        # Update positions for the next iteration
        for i in range(self.pop_size):
            if best_next_nodes[i] == self.positions[i]:
                self.positions[i] = random.randint(0, self.num_nodes - 1)
            else:
                self.positions[i] = best_next_nodes[i]
                
        # Track global best
        current_best_idx = np.argmin(self.ant_costs)
        current_best_cost = self.ant_costs[current_best_idx]
        
        if current_best_cost < self.global_best_score:
            self.global_best_score = current_best_cost
            self.global_best_position = self.positions[current_best_idx]