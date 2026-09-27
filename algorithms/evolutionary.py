import numpy as np
import scipy.special as sc
from heuristic_opt.base import BaseContinuousOptimizer

class CSO(BaseContinuousOptimizer):
    """Cuckoo Search Optimization"""
    def __init__(self, obj_func, dim, bounds, pop_size=25, max_iter=200, pa=0.25, beta=1.5):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter)
        self.pa = pa
        self.beta = beta
        self.sigma = (sc.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                     (sc.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)

    def step(self, t):
        # 1. Levy Flights
        u = np.random.normal(0, self.sigma, (self.pop_size, self.dim))
        v = np.random.normal(0, 1, (self.pop_size, self.dim))
        step = u / (np.abs(v) ** (1 / self.beta))
        
        new_positions = self.positions + 0.01 * step
        new_positions = np.clip(new_positions, self.lb, self.ub)
        new_costs = np.apply_along_axis(self.objective_function, 1, new_positions)
        
        improved = new_costs < self.costs
        self.positions[improved] = new_positions[improved]
        
        # 2. Abandon worst nests
        abandon_mask = np.random.rand(self.pop_size) < self.pa
        step_random = np.random.rand(self.pop_size, self.dim) * (self.positions[np.random.permutation(self.pop_size)] - self.positions)
        self.positions[abandon_mask] += step_random[abandon_mask]

class EHO(BaseContinuousOptimizer):
    """Elephant Herding Optimization"""
    def step(self, t):
        alpha, beta = 0.7, 0.7
        best_idx, worst_idx = np.argmin(self.costs), np.argmax(self.costs)
        center = np.mean(self.positions, axis=0)
        
        for i in range(self.pop_size):
            if i == best_idx:
                self.positions[i] = beta * center
            elif i == worst_idx:
                self.positions[i] = np.random.uniform(self.lb, self.ub, self.dim)
            else:
                self.positions[i] += alpha * (self.positions[best_idx] - self.positions[i]) * np.random.rand(self.dim)

class DFO(BaseContinuousOptimizer):
    """Dispersive Fly Optimization"""
    def step(self, t):
        delta = 0.1
        for i in range(self.pop_size - 1):
            neighbor_best = self.positions[i-1] if self.costs[i-1] < self.costs[i+1] else self.positions[i+1]
            if np.random.rand() < delta:
                self.positions[i] = np.random.uniform(self.lb, self.ub, self.dim)
            else:
                self.positions[i] = neighbor_best + np.random.rand() * (self.global_best_position - self.positions[i])

class HUS(BaseContinuousOptimizer):
    """Hunter Search"""
    def step(self, t):
        MML, HGCR, alpha, beta, EN = 0.2, 0.3, 2, 5, 1
        R = 0.5 * (self.ub - self.lb) * np.exp(-t / self.max_iter)
        
        for i in range(self.pop_size):
            # Move to leader
            self.positions[i] += np.random.rand() * MML * (self.global_best_position - self.positions[i])
            # Cooperation
            if np.random.rand() > HGCR:
                self.positions[i] += R if np.random.rand() < 0.5 else -R
            # Reorganization
            if np.abs(self.global_best_score - np.max(self.costs)) < 0.01:
                self.positions[i] = self.global_best_position + np.random.rand() * (self.ub - self.lb) * alpha * np.exp(beta * EN)

class MBO(BaseContinuousOptimizer):
    """Monarch Butterfly Optimization"""
    def step(self, t):
        p_ratio = 5/12
        split = int(self.pop_size * p_ratio)
        sorted_idx = np.argsort(self.costs)
        
        for i in range(self.pop_size):
            if i < split: # Region 1 Migration
                if np.random.rand() <= p_ratio:
                    self.positions[i] = self.positions[np.random.randint(0, split)]
            else: # Region 2 Adjusting
                if np.random.rand() < p_ratio:
                    self.positions[i] = self.positions[sorted_idx[0]]
                else:
                    self.positions[i] += (1.0 / ((t + 1)**2)) * np.random.randn(self.dim)

class BFO(BaseContinuousOptimizer):
    """Bacterial Foraging Optimization"""
    def __init__(self, obj_func, dim, bounds, pop_size=20, N_c=20, N_re=10, N_ed=10, P_ed=0.02, C=0.25):
        super().__init__(obj_func, dim, bounds, pop_size, N_c * N_re * N_ed)
        self.N_c, self.N_re, self.N_ed = N_c, N_re, N_ed
        self.P_ed, self.C = P_ed, C

    def step(self, t):
        pass # Overridden by custom optimize loop

    def optimize(self):
        for l in range(self.N_ed):
            for k in range(self.N_re):
                J_health = np.zeros(self.pop_size)
                for j in range(self.N_c):
                    for i in range(self.pop_size):
                        delta = np.random.uniform(-1, 1, self.dim)
                        tumble = delta / np.linalg.norm(delta)
                        self.positions[i] += self.C * tumble
                        self.enforce_bounds()
                        self.costs[i] = self.objective_function(self.positions[i])
                        J_health[i] += self.costs[i]
                
                # Reproduction: Sort by health, replace worst half with best half
                sorted_idx = np.argsort(J_health)
                half = self.pop_size // 2
                for i in range(half, self.pop_size):
                    self.positions[sorted_idx[i]] = self.positions[sorted_idx[i - half]].copy()
            
            # Elimination-Dispersal
            for i in range(self.pop_size):
                if np.random.rand() < self.P_ed:
                    self.positions[i] = np.random.uniform(self.lb, self.ub, self.dim)
            
            self.update_costs_and_bests(np.apply_along_axis(self.objective_function, 1, self.positions))
        return self.global_best_position, self.global_best_score