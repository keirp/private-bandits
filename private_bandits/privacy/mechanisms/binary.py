import math
import numpy as np

class BinaryMechanism:
    
    def __init__(self, T, epsilon):
        self.bits = math.ceil(np.log2(T))
        self.epsilon_prime_inv = self.bits / epsilon
        self.t = 0
        self.alpha = [0] * self.bits
        self.alpha_hat = [0] * self.bits
        
    def step(self, item):
        self.t += 1
        # Estimates the sum of items up to this point.
        ith_bit = lambda a, i: (a >> i) & 1
        i = int(np.log2((self.t & ((~self.t) + 1))))
        self.alpha[i] = sum(self.alpha[:i]) + item
        for j in range(i):
            self.alpha[j] = 0
            self.alpha_hat[j] = 0
        self.alpha_hat[i] = self.alpha[i] + np.random.laplace(0, self.epsilon_prime_inv)
        activated_x = list(filter(lambda x: ith_bit(self.t, x) == 1, range(self.bits)))
        return sum([self.alpha_hat[x] for x in activated_x])