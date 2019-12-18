from private_bandits.solvers.counting import EquitablePrivateMechanismCountingSolver, PrivateMechanismCountingSolver
from private_bandits.privacy.mechanisms.binary import BinaryMechanism
import numpy as np
import math

class DPBinarySoftmaxSolver(EquitablePrivateMechanismCountingSolver):
    
    def __init__(self, n_arms, epsilon, T):
        super().__init__(n_arms, epsilon, T, mech_cls=BinaryMechanism)
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
        temp = np.sqrt(1. / min(self.counts))
        return self._softmax(self.estimates, temp)
    
    def __str__(self):
        return f'DPBinarySoftmax(epsilon={self.epsilon})'