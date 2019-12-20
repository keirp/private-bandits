import numpy as np
import multiprocessing
from tqdm import tqdm

class Runner:
    
    def __init__(self, bandits, solvers):
        self.bandits = bandits
        self.solvers = solvers
        
    def run_all(self, n_steps):
        last_reward = [None] * len(self.solvers)
        last_arm = [None] * len(self.solvers)
        exp_returns = [0] * len(self.solvers)
        best_possible = [0] * len(self.solvers)
        regrets = []
        for _ in range(n_steps):
            for i, solver in enumerate(self.solvers):
                arm = solver.step(last_reward[i], last_arm[i])
                last_reward[i], optimal, exp_return = self.bandits.pull_arm(arm)
                last_arm[i] = arm
                exp_returns[i] += exp_return
                best_possible[i] += optimal
            regret = [best_possible[i] - exp_returns[i] for i in range(len(self.solvers))]
            regrets.append(regret)
        return regrets
    
class MultiRunner:
    
    def __init__(self, bandits_cls, solvers_cls):
        self.bandits_cls = bandits_cls
        self.solvers_cls = solvers_cls
        
    def reset(self):
        bandits = self.bandits_cls()
        solvers = [x() for x in self.solvers_cls]
        self.runner = Runner(bandits, solvers)
        
    def run_all(self, n_steps, n_runs):
        multi_regrets = []
        for run in tqdm(range(n_runs)):
            self.reset()
            regrets = self.runner.run_all(n_steps)
            multi_regrets.append(regrets)
        multi_regrets = np.array(multi_regrets)
        multi_regrets = np.sum(multi_regrets, axis=0) / n_runs
        return multi_regrets
    
    def multiprocess_run(self, n_steps):
        bandits = self.bandits_cls()
        solvers = [x() for x in self.solvers_cls]
        runner = Runner(bandits, solvers)
        return runner.run_all(n_steps)