import numpy as np

class BernoulliBandits:
    
    def __init__(self, n_arms, probs=None):
        self.n_arms = n_arms
        if probs is None:
            self.probs = [np.random.random() for _ in range(self.n_arms)]
        else:
            self.probs = probs
        rounded_probs = np.round(self.probs, decimals=2)
#         print(f'Actual bandits params: {rounded_probs}')
        self.max_prob = max(self.probs)
                
    def pull_arm(self, i):
        rewarded = np.random.random() < self.probs[i]
        if rewarded:
            reward = 1
        else:
            reward = 0
        return reward, self.max_prob, self.probs[i]