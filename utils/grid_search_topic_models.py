#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
grid search a list of hyperparameters for topic modeling
"""

import os, itertools
import pandas as pd
import tomotopy as tp
from tomotopy.utils import Corpus
from tomotopy.coherence import Coherence

# load the files
indir = "../data/session_three/input/"
fnames = os.listdir(indir)
paths = [indir + fname for fname in fnames if fname.startswith(".") == False]

# build a tomotopy corpus
corpus = Corpus()
for path in paths:
    with open(path, 'r') as p:
        doc = p.read().split()
        corpus.add_doc(doc)

# set a range of hyperparameters
k = list(range(5, 31, 5))
alphas = [0.001, 0.1, 10]
etas = [0.001, 0.1, 10]
params = list(itertools.product(k, alphas, etas))
print("Parameter combinations to search:", len(params))

# roll through all possible combinations
results = []
for idx, param in enumerate(params):
    n_topics, alpha, eta = param[0], param[1], param[2]
    model = tp.LDAModel(
                k = n_topics,
                alpha = alpha,
                eta = eta,
                corpus = corpus,
                seed = 357
                )
    model.train(iter = 1000)
    perplexity = model.perplexity
    coherence_model = Coherence(model, coherence = 'c_v')
    coherence = coherence_model.get_score()
    results.append({
                'N_TOPICS': n_topics,
                'ALPHA': alpha,
                'ETA': eta,
                'PERPLEXITY': perplexity,
                'COHERENCE': coherence
                })
    # tiny bit of progress logging
    if idx % 10 == 0:
        print(f"Completed {idx} of {len(params)} combinations")

# compile the results and save them
results = pd.DataFrame(results)
results.to_csv("../data/session_three/grid_search_model_results.csv")

