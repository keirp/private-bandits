from private_bandits.solvers.counting import CountingSolver
import math
import numpy as np

class DPSoftmaxSolver(CountingSolver):
    
    def __init__(self, n_arms, epsilon, T, init_estimate=1.0):
        super().__init__(n_arms, init_estimate)
        self.epsilon = epsilon
        self.epsilon_zero = self.epsilon / T
        
    def _softmax(self, x, temp):
        exps = [math.exp(x[i] / temp) for i in range(len(x))]
        sum_exp = sum(exps)
        dist = [exps[i] / sum_exp for i in range(len(x))]
        return np.random.choice(range(len(x)), p=dist)
        
    def _sensitivity(self):
        min_count = min(self.counts)
        return 1. / min_count
        
    def step(self, last_reward, last_arm):
        super().step(last_reward, last_arm)
        for i, count in enumerate(self.counts):
                if count == 0:
                    return i
        temp = (2. * self._sensitivity()) / self.epsilon_zero
        return self._softmax(self.estimates, temp)
        
    def __str__(self):
        return f'DPSoftmax(epsilon={self.epsilon})'