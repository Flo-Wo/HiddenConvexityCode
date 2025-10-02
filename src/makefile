# Reproduce figures in the paper, e.g. with
# execute in ``src/`` the command
# make figure_1_ab

# Section 1: Introduction
figure_1_ab:
	python 1_nonsmooth_example/main.py
figure_1_cd:
	python 2_smooth_example/main.py 
# Section 2: Hidden Convex Problem class
figure_2:
	python 3_nonsmooth_consistency/main.py
# Section 4: Bundle-level
figure_3_a_c:
	python 5_PMM_objective_illustration/main.py
figure_3_b:
	python 4_PMM_constraint_illustration/main.py
# Section 5: Numerical Experiments
figure_4:
	python 1_nonsmooth_example/main.py --run_algos
figure_5:
	python 2_smooth_example/main.py  --run_algos
figure_6:
	python 6_CGP_high_dim/main.py
