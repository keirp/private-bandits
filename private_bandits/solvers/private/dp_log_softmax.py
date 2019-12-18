from private_bandits.solvers.counting import EquitablePrivateMechanismCountingSolver, PrivateMechanismCountingSolver
from private_bandits.privacy.mechanisms.logarithmic import LogarithmicMechanism
import numpy as np
import math

class DPLogSoftmaxSolver(PrivateMechanismCountingSolver):
    
    def __init__(self, n_arms, epsilon, T):
        super().__init__(n_arms, epsilon, None, mech_cls=LogarithmicMechanism)
        self.T = T

    def _softmax(self, x, temp):
        x_bar = [x[i] / temp for i in range(len(x))]
        max_x_bar = max(x_bar)
        x_bar = [y - max_x_bar for y in x_bar]
        exps = [math.exp(x_bar[i]) for i in range(len(x))]
        sum_exp = sum(exps)
        dist = [exps[i] / sum_exp for i in range(len(x))]
        return np.random.choice(range(len(x)), p=dist)
        
    def step(self, last_reward, last_arm):
        super().step(last_reward, last_arm)
        for i, count in enumerate(self.counts):
                if count == 0:
                    return i
        num_noise = np.floor(np.log2(min(self.counts))) + 1
        var = (2 * num_noise) / (self.epsilon ** 2)
        std_dev = np.sqrt(var) / (2 ** num_noise)
        temp = np.sqrt(1. / min(self.counts)) + std_dev
        return self._softmax(self.estimates, temp)
    
    def __str__(self):
        return f'DPLogSoftmax(epsilon={self.epsilon})'