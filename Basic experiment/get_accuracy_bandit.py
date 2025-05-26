import numpy as np
import pandas as pd
'''
this function takes the path to a csv file and the best arm indexes as input
it returns the probability of each arm being the best arm at each time point
Used for comparison with the experimental results in this paper.
'''

def get_accuracy_bandit(fp, best_arm_indexes):
    df = pd.read_csv(fp)
    df = df[['num_sims', 'horizon', 'chosen_arm']]
    n_simulations = int(np.max(df['num_sims'])) + 1
    time_horizon = int(np.max(df['horizon'])) + 1
    best_arms = np.zeros((n_simulations, time_horizon)) 
    for n in range(int(n_simulations)):
        data = np.array(list(df.loc[df['num_sims'] == n]['chosen_arm']))
        for t in range(len(data)):
            u, counts = np.unique(data[:t+1], return_counts=True)
            best_arms[n, t] = u[np.random.choice(np.flatnonzero(counts == max(counts)))]
    isinfunc = lambda x: x in best_arm_indexes
    visinfunc = np.vectorize(isinfunc)
    boo = visinfunc(best_arms)
    probs = boo.sum(axis=0)/n_simulations
    return probs