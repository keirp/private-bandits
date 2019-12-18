from private_bandits.solvers.counting import CountingSolver
import numpy as np

class EpsilonGreedySolver(CountingSolver):
    
    def __init__(self, epsilon, n_arms, init_estimate=1.0):
        self.epsilon = epsilon
        super().__init__(n_arms, init_estimate)
        
    def step(self, last_reward, last_arm):
        super().step(last_reward, last_arm)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        return max(range(self.n_arms), key=lambda x: self.estimates[x])
    
    def __str__(self):
        return f'EpsilonGreedy(epsilon={self.epsilon})'