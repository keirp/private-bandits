from private_bandits.solvers.base import Solver

class CountingSolver(Solver):
    
    def __init__(self, n_arms, init_estimate=1.0):
        self.n_arms = n_arms
        self.estimates = [init_estimate] * self.n_arms
        self.counts = [0] * self.n_arms
        
    def step(self, last_reward, last_arm):
        if last_reward is not None:
            self.estimates[last_arm] += 1. / (self.counts[last_arm] + 1) * (last_reward - self.estimates[last_arm])
            self.counts[last_arm] += 1

class PrivateMechanismCountingSolver(Solver):
    
    def __init__(self, n_arms, epsilon, T, mech_cls):
        self.n_arms = n_arms
        self.T = T
        self.counts = [0] * self.n_arms
        self.epsilon = epsilon
#         self.epsilon_zero = self.epsilon / self.n_arms
        self.epsilon_zero = self.epsilon
        self.summers = [mech_cls(self.T, self.epsilon_zero) for _ in range(self.n_arms)]
        self.estimates = [None] * self.n_arms
        
    def step(self, last_reward, last_arm):
        if last_reward is not None:
            self.counts[last_arm] += 1
            estimate = self.summers[last_arm].step(last_reward)
            if estimate is not None:
                self.estimates[last_arm] = estimate / self.counts[last_arm]

class EquitablePrivateMechanismCountingSolver(Solver):
    
    def __init__(self, n_arms, epsilon, T, mech_cls):
        self.n_arms = n_arms
        self.T = T
        self.counts = [0] * self.n_arms
        self.epsilon = epsilon
#         self.epsilon_zero = self.epsilon / self.n_arms
        self.epsilon_zero = self.epsilon
        self.summers = [mech_cls(self.T, self.epsilon_zero) for _ in range(self.n_arms)]
        self.estimates = [None] * self.n_arms
        
    def step(self, last_reward, last_arm):
        if last_reward is not None:
            self.counts[last_arm] += 1
            rewards = [0 for i in range(self.n_arms)]
            rewards[last_arm] = last_reward
            for a in range(self.n_arms): 
                estimate = self.summers[a].step(rewards[a])
                if self.counts[a] > 0 and estimate is not None:
                    self.estimates[a] = estimate / self.counts[a]