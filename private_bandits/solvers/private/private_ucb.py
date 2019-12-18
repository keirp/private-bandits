from private_bandits.solvers.counting import PrivateMechanismCountingSolver
from private_bandits.privacy.mechanisms.binary import BinaryMechanism
import numpy as np

class PrivateUCBSolver(PrivateMechanismCountingSolver):
    
        def __init__(self, n_arms, epsilon, T, gamma):
            super().__init__(n_arms, epsilon, T, mech_cls=BinaryMechanism)
            self.gamma = gamma
    
        def confidence_relaxation(self):
            return (self.n_arms * (np.log(self.T) ** 2) * np.log((self.n_arms * self.T * np.log(self.T)) / self.gamma)) / self.epsilon
        
        def step(self, last_reward, last_arm):
            super().step(last_reward, last_arm)
            for i, count in enumerate(self.counts):
                if count == 0:
                    return i
            t = sum(self.counts)
            c_r = self.confidence_relaxation()
            ucb_range = [np.sqrt((2. * np.log(t)) / self.counts[i]) + (c_r / self.counts[i]) for i in range(self.n_arms)]
            ucb = [self.estimates[i] + ucb_range[i] for i in range(self.n_arms)]
            return max(range(self.n_arms), key=lambda x: ucb[x])
    
        def __str__(self):
            return f'PrivateUCB(epsilon={self.epsilon}, gamma={self.gamma})'