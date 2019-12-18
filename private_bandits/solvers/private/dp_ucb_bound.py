from private_bandits.solvers.counting import PrivateMechanismCountingSolver
from private_bandits.privacy.mechanisms.hybrid import HybridMechanism
import numpy as np
import math

class DPUCBBoundSolver(PrivateMechanismCountingSolver):
    
    def __init__(self, n_arms, epsilon):
        super().__init__(n_arms, epsilon, None, mech_cls=HybridMechanism)
        
    def step(self, last_reward, last_arm):
        super().step(last_reward, last_arm)
        for i, count in enumerate(self.counts):
                if count == 0:
                    return i
        t = sum(self.counts)
        v_a = []
        const = (math.sqrt(8.) * math.log(4. * (t ** 4))) / self.epsilon
        for a in range(self.n_arms):
            if math.log2(self.counts[a]).is_integer():
                v_a.append(const)
            else:
                v_a.append(const * math.log(self.counts[a]) + const)
        ucb_range = [np.sqrt((2. * np.log(t)) / self.counts[i]) for i in range(self.n_arms)]
        ucb = [self.estimates[i] + ucb_range[i] + (v_a[i] / self.counts[i]) for i in range(self.n_arms)]
        return max(range(self.n_arms), key=lambda x: ucb[x])
    
    def __str__(self):
        return f'DPUCBBound(epsilon={self.epsilon})'