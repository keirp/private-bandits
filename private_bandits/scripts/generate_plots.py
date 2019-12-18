from private_bandits.solvers.nonprivate.epsilon_greedy import EpsilonGreedySolver
from private_bandits.solvers.nonprivate.softmax import SoftmaxSolver
from private_bandits.solvers.nonprivate.ucb import UCB1Solver

from private_bandits.solvers.private.dp_softmax import DPSoftmaxSolver
from private_bandits.solvers.private.dp_ucb import DPUCBSolver
from private_bandits.solvers.private.dp_ucb_bound import DPUCBBoundSolver
from private_bandits.solvers.private.private_ucb import PrivateUCBSolver
from private_bandits.solvers.private.private_ucb_boundless import PrivateUCBBoundlessSolver
from private_bandits.solvers.private.dp_successive_elimination import DPSuccessiveEliminationSolver
from private_bandits.solvers.private.dp_log_softmax import DPLogSoftmaxSolver
from private_bandits.solvers.private.dp_binary_softmax import DPBinarySoftmaxSolver

from private_bandits.bandits.bernoulli_bandits import BernoulliBandits
from private_bandits.bandits.runners import MultiRunner

from absl import app
from absl import flags

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_arms', 2, 'Number of bandits arms')
flags.DEFINE_float('p_epsilon', 1, 'Privacy epsilon')
flags.DEFINE_integer('n_runs', 50, 'Number of runs to average for the final result')
flags.DEFINE_integer('T', 1000, 'Number of steps to run for')
flags.DEFINE_list('probs', None, 'Arm reward probabilities')
flags.DEFINE_bool('log', False, 'Plot with log scale')

def generate_experiment_data(n_arms, probs, p_epsilon, T, n_runs):
	bandits = lambda: BernoulliBandits(n_arms=n_arms, probs=probs)

	eps_greedy_solver = lambda: EpsilonGreedySolver(epsilon=0.01, n_arms=n_arms)
	ucb_solver = lambda: UCB1Solver(n_arms=n_arms)
	softmax_solver = lambda: SoftmaxSolver(n_arms=n_arms, T=T)

	private_ucb_solver = lambda: PrivateUCBSolver(n_arms=n_arms, epsilon=p_epsilon, T=T, gamma=T ** -4)
	private_ucb_boundless_solver = lambda: PrivateUCBBoundlessSolver(n_arms=n_arms, T=T, epsilon=p_epsilon)
	dp_ucb_bound_solver = lambda: DPUCBBoundSolver(n_arms=n_arms, epsilon=p_epsilon)
	dp_ucb_solver = lambda: DPUCBSolver(n_arms=n_arms, epsilon=p_epsilon)

	dp_softmax_solver = lambda: DPSoftmaxSolver(n_arms=n_arms, epsilon=p_epsilon, T=T)

	dp_successive_elimination_solver = lambda: DPSuccessiveEliminationSolver(n_arms=n_arms, epsilon=p_epsilon, beta=T ** -1)

	dp_log_softmax_solver = lambda: DPLogSoftmaxSolver(n_arms=n_arms, epsilon=p_epsilon, T=T)
	dp_binary_softmax_solver = lambda: DPBinarySoftmaxSolver(n_arms=n_arms, epsilon=p_epsilon, T=T)

	runner = MultiRunner(bandits, [eps_greedy_solver,
                               softmax_solver,
                               ucb_solver,
                               private_ucb_solver,
                               private_ucb_boundless_solver,
                               dp_softmax_solver,
                               dp_ucb_solver,
                               dp_ucb_bound_solver,
                               dp_successive_elimination_solver,
                               dp_log_softmax_solver,
                               dp_binary_softmax_solver])

	# runner = MultiRunner(bandits, [dp_log_softmax_solver, dp_binary_softmax_solver, private_ucb_boundless_solver])

	return runner.run_all(T, n_runs), [str(c()) for c in runner.solvers_cls]

def main(argv):
	if FLAGS.probs is not None:
		probs = [float(x) for x in FLAGS.probs]
	else:
		probs = None
	regrets, labels = generate_experiment_data(FLAGS.n_arms, probs, FLAGS.p_epsilon, FLAGS.T, FLAGS.n_runs)

	if FLAGS.log:
		plt.yscale('log')

	for i in range(regrets.shape[1]):
	    lw = 8-6*i/regrets.shape[1]
	    ls = ['-','--','-.'][i%3]
	    solver = labels[i]
	    plt.plot(regrets[:, i], label=solver, linewidth=lw, linestyle=ls)

	plt.grid(True)
	plt.xlabel('Steps', fontsize=20)
	plt.ylabel('Regret', fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.title(f'{FLAGS.n_arms} arms (Averaged over {FLAGS.n_runs} runs)', fontsize=30)
	plt.legend(prop={'size': 20})
	plt.savefig(f'private_bandits/results/{FLAGS.n_arms}_arms_{FLAGS.p_epsilon}_eps_{FLAGS.T}_T_{FLAGS.probs is not None}_preset_probs.png')


if __name__ == '__main__':
	app.run(main)