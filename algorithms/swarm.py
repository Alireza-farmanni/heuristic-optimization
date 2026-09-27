import numpy as np
import math
from heuristic_opt.base import BaseContinuousOptimizer

class PSO(BaseContinuousOptimizer):
    """Particle Swarm Optimization"""
    def __init__(self, obj_func, dim, bounds, pop_size=30, max_iter=100, c1=2.0, c2=2.0, w=0.7):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter)
        self.c1, self.c2, self.w = c1, c2, w
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.pbest_pos = self.positions.copy()
        self.pbest_costs = self.costs.copy()

    def update_costs_and_bests(self, new_costs):
        improved = new_costs < self.pbest_costs
        self.pbest_pos[improved] = self.positions[improved]
        self.pbest_costs[improved] = new_costs[improved]
        super().update_costs_and_bests(new_costs)

    def step(self, t):
        r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
        self.velocities = (self.w * self.velocities + 
                           self.c1 * r1 * (self.pbest_pos - self.positions) + 
                           self.c2 * r2 * (self.global_best_position - self.positions))
        self.positions += self.velocities

class GWO(BaseContinuousOptimizer):
    """Grey Wolf Optimizer"""
    def step(self, t):
        a = 2.0 - t * (2.0 / self.max_iter)
        sorted_idx = np.argsort(self.costs)
        alpha, beta, delta = self.positions[sorted_idx[0]], self.positions[sorted_idx[1]], self.positions[sorted_idx[2]]
        
        r1, r2 = np.random.rand(3, self.pop_size, self.dim), np.random.rand(3, self.pop_size, self.dim)
        A, C = 2 * a * r1 - a, 2 * r2
        
        X1 = alpha - A[0] * np.abs(C[0] * alpha - self.positions)
        X2 = beta - A[1] * np.abs(C[1] * beta - self.positions)
        X3 = delta - A[2] * np.abs(C[2] * delta - self.positions)
        
        self.positions = (X1 + X2 + X3) / 3.0

class SSA(BaseContinuousOptimizer):
    """Salp Swarm Algorithm"""
    def step(self, t):
        c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))
        c2 = np.random.rand(self.pop_size, self.dim)
        c3 = np.random.choice([-1, 1], size=(self.pop_size, self.dim))
        
        # Leader update (first agent)
        self.positions[0] = self.global_best_position + c1 * ((self.ub - self.lb) * c2[0] + self.lb) * c3[0]
        
        # Follower updates
        for i in range(1, self.pop_size):
            self.positions[i] = (self.positions[i] + self.positions[i-1]) / 2.0

class BA(BaseContinuousOptimizer):
    """Bat Algorithm"""
    def __init__(self, obj_func, dim, bounds, pop_size=30, max_iter=100, f_min=0, f_max=1, alpha=0.9, gamma=0.9):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter)
        self.f_min, self.f_max = f_min, f_max
        self.alpha, self.gamma = alpha, gamma
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.A = np.full(self.pop_size, 0.5)
        self.r = np.full(self.pop_size, 0.1)
        self.r_initial = 0.1

    def step(self, t):
        beta = np.random.rand(self.pop_size, 1)
        freq = self.f_min + (self.f_max - self.f_min) * beta
        self.velocities += (self.positions - self.global_best_position) * freq
        new_positions = self.positions + self.velocities

        # Local search based on pulse rate
        for i in range(self.pop_size):
            if np.random.rand() > self.r[i]:
                new_positions[i] = self.global_best_position + 0.001 * np.random.randn(self.dim) * np.mean(self.A)
            
            if np.random.rand() < self.A[i] and self.objective_function(new_positions[i]) < self.costs[i]:
                self.positions[i] = new_positions[i]
                self.A[i] *= self.alpha
                self.r[i] = self.r_initial * (1 - math.exp(-self.gamma * t))

class CSA(BaseContinuousOptimizer):
    """Crow Search Algorithm"""
    def __init__(self, obj_func, dim, bounds, pop_size=30, max_iter=100, fl=2.0, ap=0.1):
        super().__init__(obj_func, dim, bounds, pop_size, max_iter)
        self.fl, self.ap = fl, ap
        self.memory = self.positions.copy()
        self.memory_costs = self.costs.copy()

    def update_costs_and_bests(self, new_costs):
        improved = new_costs < self.memory_costs
        self.memory[improved] = self.positions[improved]
        self.memory_costs[improved] = new_costs[improved]
        super().update_costs_and_bests(new_costs)

    def step(self, t):
        targets = np.random.randint(0, self.pop_size, self.pop_size)
        r = np.random.rand(self.pop_size, 1)
        awareness_mask = np.random.rand(self.pop_size, 1) > self.ap
        
        # Flight towards target memory
        flight = self.positions + r * self.fl * (self.memory[targets] - self.positions)
        # Random relocation if awareness is triggered
        random_pos = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        self.positions = np.where(awareness_mask, flight, random_pos)

class GOA(BaseContinuousOptimizer):
    """Grasshopper Optimization Algorithm"""
    def step(self, t):
        c_max, c_min = 1.0, 0.00001
        c = c_max - t * ((c_max - c_min) / self.max_iter)
        
        for i in range(self.pop_size):
            S_i = np.zeros(self.dim)
            for j in range(self.pop_size):
                if i != j:
                    dist = np.linalg.norm(self.positions[j] - self.positions[i])
                    if dist > 0:
                        s_val = 0.5 * math.exp(-dist / 1.5) - math.exp(-dist)
                        direction = (self.positions[j] - self.positions[i]) / dist
                        S_i += c * s_val * direction
            self.positions[i] = c * S_i + self.global_best_position

class KH(BaseContinuousOptimizer):
    """Krill Herding (Simplified Foraging & Diffusion)"""
    def step(self, t):
        V_f, D_max = 0.02, 0.005
        omega = 0.1 + 0.8 * (1 - t / self.max_iter)
        
        # Center of mass calculation
        com = np.average(self.positions, axis=0, weights=1.0 / (self.costs + 1e-5))
        
        for i in range(self.pop_size):
            foraging = V_f * (self.positions[i] * com + self.global_best_position * self.global_best_score)
            diffusion = D_max * (1 - t / self.max_iter) * np.random.randn(self.dim)
            self.positions[i] += omega * foraging + diffusion