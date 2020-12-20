import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from PyExpUtils.utils.Collector import Collector
from src.utils.rlglue import OneStepWrapper

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

max_steps = exp.max_steps

collector = Collector()
broke = False
best = -np.inf
for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)
    print('        ', end='\r')
    print(run, end='\r')

    inner_idx = exp.numPermutations() * run + idx
    Problem = getProblem(exp.problem)
    problem = Problem(exp, inner_idx, idx)

    agent = problem.getAgent()
    env = problem.getEnvironment()

    wrapper = OneStepWrapper(agent, problem.getGamma(), problem.rep)

    glue = RlGlue(wrapper, env)

    # Run the experiment
    for episode in range(exp.episodes):
        glue.total_reward = 0
        glue.num_steps = 0
        glue.runEpisode(max_steps)

        collector.collect('return', glue.total_reward)


    average = np.mean(collector.run_data['return'])
    collector.collect('average', average)

    collector.reset()

    if average > best:
        best = average
        print(run, average)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1)

min_idx = np.argmin(collector.all_data['average'])
max_idx = np.argmax(collector.all_data['average'])

min_run = collector.all_data['return'][min_idx]
max_run = collector.all_data['return'][max_idx]

lo = np.min([min_run, max_run])
hi = np.max([min_run, max_run])

min_hist, _ = np.histogram(min_run, bins=20, range=(lo, hi))
max_hist, _ = np.histogram(max_run, bins=20, range=(lo, hi))

hist_x = np.linspace(lo, hi, 20)
ax1.bar(hist_x, min_hist, width=(hi - lo)/20, color='red', label='worst', alpha=0.6)
ax1.bar(hist_x, max_hist, width=(hi - lo)/20, color='blue', label='best', alpha=0.6)

ax1.set_title('Cartpole')
ax1.set_ylabel('Number of episodes')
ax1.set_xlabel('Return')

plt.show()
