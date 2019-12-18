from private_bandits.solvers.base import Solver
from private_bandits.privacy.mechanisms.hybrid import HybridMechanism
import numpy as np

class DPSuccessiveEliminationSolver(Solver):
    
    def __init__(self, n_arms, epsilon, beta):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.beta = beta
        self.epoch = 0
        self.S = list(range(self.n_arms))
        self._new_epoch()

    def _new_epoch(self):
        self.epoch += 1
        self.r = 0
        self.estimates = [0] * self.n_arms
        self.counts = [0] * self.n_arms
        self.delta_e = 2. ** (-self.epoch)
        a = (32. * np.log(8. * len(self.S) * (self.epoch ** 2) / self.beta)) / (self.delta_e ** 2)
        b = (8. * np.log(4 * len(self.S) * (self.epoch ** 2) / self.beta)) / (self.delta_e * self.epsilon)
        self.R_e = max(a, b) + 1.
        self.arm_counter = 0

    def step(self, last_reward, last_arm):

        # There is only one arm left, so pull it.
        if len(self.S) == 1:
            return self.S[0]

        if last_reward is not None:
            # update our mean estimation
            if self.counts[last_arm] == 0:
                self.estimates[last_arm] = last_reward
            else:
                self.estimates[last_arm] += 1. / (self.counts[last_arm] + 1) * (last_reward - self.estimates[last_arm])
            self.counts[last_arm] += 1

            # move onto the next arm (unless it is time to start back at 0)
            self.arm_counter += 1
            if self.arm_counter >= len(self.S):
                self.r += 1
                self.arm_counter = 0

            # check if we have pulled the arms enough times
            if self.r >= self.R_e:
                h_e = np.sqrt(np.log(8. * len(self.S) * (self.epoch ** 2) / self.beta)/(2. * self.R_e))
                c_e = np.log(4. * len(self.S) * (self.epoch ** 2) / self.beta) / (self.R_e * self.epsilon)
                noisy_estimates = [mu + np.random.laplace(0, 1. / (self.epsilon * self.r)) for mu in self.estimates]
                mu_max = float('-inf')
                for i in self.S:
                    if noisy_estimates[i] > mu_max:
                        mu_max = noisy_estimates[i]

                # eliminate all bad arms
                self.S = [s for s in self.S if mu_max - noisy_estimates[s] <= 2. * h_e + 2. * c_e]
                self._new_epoch()
        
        return self.S[self.arm_counter]

    
    def __str__(self):
        return f'DPSuccessiveElimination(epsilon={self.epsilon}, beta={self.beta})'