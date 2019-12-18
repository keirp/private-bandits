from private_bandits.solvers.counting import EquitablePrivateMechanismCountingSolver
from private_bandits.privacy.mechanisms.hybrid import HybridMechanism
import numpy as np

class DPUCBSolver(EquitablePrivateMechanismCountingSolver):
    
    def __init__(self, n_arms, epsilon):
        super().__init__(n_arms, epsilon, None, mech_cls=HybridMechanism)
        
    def step(self, last_reward, last_arm):
        super().step(last_reward, last_arm)
        for i, count in enumerate(self.counts):
                if count == 0:
                    return i
        t = sum(self.counts)
        ucb_range = [np.sqrt((2. * np.log(t)) / self.counts[i]) for i in range(self.n_arms)]
        ucb = [self.estimates[i] + ucb_range[i] for i in range(self.n_arms)]
        return max(range(self.n_arms), key=lambda x: ucb[x])
    
    def __str__(self):
        return f'DPUCB(epsilon={self.epsilon})'