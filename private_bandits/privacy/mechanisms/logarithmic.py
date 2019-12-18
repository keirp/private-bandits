import math
import numpy as np

class LogarithmicMechanism:
    
    def __init__(self, T, epsilon):
        self.eps_inv = 1. / epsilon
        self.beta = 0
        self.t = 0
        
    def step(self, item):
        self.beta += item
        self.t += 1
        if math.log2(self.t).is_integer():
            self.beta += np.random.laplace(0, self.eps_inv)
            return self.beta
        return None