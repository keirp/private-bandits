python private_bandits/scripts/generate_plots.py --log=False --n_runs=50 --probs=0.9,0.6 --p_epsilon=1 -T=1000 &
python private_bandits/scripts/generate_plots.py --log=False --n_runs=50 --probs=0.9,0.6 --p_epsilon=0.1 --T=1000 &
python private_bandits/scripts/generate_plots.py --log=True --n_runs=10 --probs=0.9,0.6 --p_epsilon=1 --T=100000 &
python private_bandits/scripts/generate_plots.py --log=True --n_runs=10 --probs=0.9,0.6 --p_epsilon=0.1 --T=100000 &
python private_bandits/scripts/generate_plots.py --log=False --n_runs=50 --probs=0.55,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 --p_epsilon=1 -T=1000 --n_arms=10 &
python private_bandits/scripts/generate_plots.py --log=False --n_runs=50 --probs=0.55,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 --p_epsilon=0.1 --T=1000 --n_arms=10 &
python private_bandits/scripts/generate_plots.py --log=True --n_runs=10 --probs=0.55,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 --p_epsilon=1 --T=100000 --n_arms=10 &
python private_bandits/scripts/generate_plots.py --log=True --n_runs=10 --probs=0.55,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 --p_epsilon=0.1 --T=100000 --n_arms=10 &
