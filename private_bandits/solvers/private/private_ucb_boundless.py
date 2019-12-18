from private_bandits.solvers.counting import EquitablePrivateMechanismCountingSolver
from private_bandits.privacy.mechanisms.binary import BinaryMechanism
import numpy as np

class PrivateUCBBoundlessSolver(EquitablePrivateMechanismCountingSolver):
    
    def __init__(self, n_arms, T, epsilon):
        super().__init__(n_arms, epsilon, T, mech_cls=BinaryMechanism)
        
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
        return f'PrivateUCBBoundless(epsilon={self.epsilon})'