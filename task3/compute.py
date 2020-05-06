'''
Requires words and features to be cached, see cache_all.py

Calculates dissimilarity using DWT and stores that into the cache folder together with features.
'''

import sys
import os
import utils
import DTW_draft
import random
import numpy as np

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
# 2. Run the trial
##################
print("STEP 2: Go over all keyWords")

n = len(train)

if not os.path.exists(utils.distance_dir):
    os.mkdir(utils.distance_dir)

for id,vfeature in valid.items():
    # if cli args set a flag - only treat words from the keywords.txt file
    if (not groundTruth[id] in keyWords) and ("-kw-only" in sys.argv): continue

    # get ground
    real_word = groundTruth[id]
    
    word_matches = len([k for k in train.keys() if groundTruth[k] == real_word])
    print(("Valid ID", id, "Ground", real_word, "Matches", word_matches))
    if (word_matches == 0): continue
    
    distances = []
    
    for tid,tfeature in train.items():
        dist = DTW_draft.DTW( #unflatten feature vectors
            vfeature.reshape((utils.dwt_feature_count,utils.dwt_image_size)),
            tfeature.reshape((utils.dwt_feature_count,utils.dwt_image_size))
            )
        distances.append([tid,dist])
    #end for

    #print(distances)
    distances.sort(key = lambda x: x[1])

    with open(utils.distance_dir + id + ".txt", "w") as f:
        f.writelines(map(lambda x: x[0] + "," + str(x[1]) + "," + groundTruth[x[0]] + "\n", distances))
        f.write(real_word)
    #end
