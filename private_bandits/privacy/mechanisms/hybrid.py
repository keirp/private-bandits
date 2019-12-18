from private_bandits.privacy.mechanisms.binary import BinaryMechanism
from private_bandits.privacy.mechanisms.logarithmic import LogarithmicMechanism

class HybridMechanism:
    
    def __init__(self, T, epsilon):
        self.half_eps = epsilon / 2.
        self.logarithmic_mech = LogarithmicMechanism(T=None, epsilon=self.half_eps)
        self.bounded_mech = None
        self.T = 1
        self.last_log_res = None
        
    def step(self, item):
        log_res = self.logarithmic_mech.step(item)
        if log_res is not None:
            self.T = self.logarithmic_mech.t
            self.last_log_res = log_res
            self.bounded_mech = BinaryMechanism(epsilon=self.half_eps, T=self.T)
            return log_res
        else:
            bounded_res = self.bounded_mech.step(item)
            return bounded_res + self.last_log_res