'''
Requires words, features and distances to be cached, see cache_all.py and compute.py

Computes P/R curves out of the distance rankings and stores the plots into the "plots" folder
'''

import sys
import os
import utils
import DTW_draft
import random
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

##############
# 1. Load data
##############
print("STEP 1: load cached data")

(train_check, valid_check) = utils.load_train_valid_pages()

(train, valid) = utils.load_cached_features() # will throw exception if not cached

print(('TrainSize', len(train)))
print(('ValidSize', len(valid)))

keyWords = utils.load_keyWords()
groundTruth = utils.load_truth()

print("ground truth loaded")


##################
# 2. Read cached files
##################
print("STEP 2: Go over all keyWords")

n = len(train)

if not os.path.exists(utils.plots_dir):
    os.mkdir(utils.plots_dir)

for id,vfeature in valid.items():
    # if cli args set a flag - only treat words from the keywords.txt file
    if (not groundTruth[id] in keyWords) and ("-kw-only" in sys.argv): continue

    # get ground
    real_word = groundTruth[id]
    
    word_matches = len([k for k in train.keys() if groundTruth[k] == real_word])
    print(("Valid ID", id, "Ground", real_word, "Matches", word_matches))
    if (word_matches == 0): continue

    if not os.path.isfile(utils.distance_dir + id + ".txt"):
        print("An entry from validation set does not have cached distances")
        print("Are you sure they were computed?")
        print("If they were computed with -kw-only flag, then plotting should be ran with it too")

    true_val = []
    scores = []

    with open(utils.distance_dir + id + ".txt", "r") as f:
        for line in f:
            if "," in line:
                l = line[:-1].split(",") #this crap has \n at the end of the line
                true_val.append(1 if l[2] == real_word else 0)
                scores.append(float(l[1]))
    # should already be sorted asc

    assert(word_matches == np.sum(true_val))
    
    scores = 1 - (scores / np.max(scores)) # uniform-normalize & invert
    (scores_prec, scores_recall, _) = precision_recall_curve(true_val, scores)
    ap = average_precision_score(true_val, scores)
    label_plt = "{}, AP:{:.2f}".format(id, ap)

    fig = plt.figure()
    prc = plt.subplot()
    prc.plot(scores_recall, scores_prec, marker='.', label=label_plt)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    prc.legend()

    plot_file = utils.plots_dir + id + ".png"
    fig.savefig(plot_file)
    plt.close(fig)
