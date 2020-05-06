'''
If not already done, this Program outputs a collection of picture of words: xxx-xx-xx.jpg  where the xxx-xx-xx is the id of the word represented the image.
Each of this picture is a portion of an original handwritten scanned page.

Then the program transforms those words images into black and white (binary), cuts off fully white rows/cols on on edges and resizes them to 100x100 pictures.

After that it extrats features for each picture.
'''

import os
import utils
import random
from PIL import Image
import numpy as np


#########################
# 1. Create word pictures
#########################
print("STEP 1: Caching words...")

if (not os.path.exists("words")) or (os.path.exists("words") and len(os.listdir("words")) < 3726):
    # i know the expression above looks strupid
    # but i dunno how it reacts to listing a non-existent dir
    # and at this point i'm too afraid to try
    # so I guarantee a boolean short-circuit
    os.mkdir("words")
    utils.convertTrainSet()
    print("train set done")
    utils.convertValidSet()
    print("valid set done")
else:
    print("words already cached")


#####################
# 2. Extract features
#####################
print("STEP 2: Extract and cache features...")

if not os.path.exists(utils.feature_cache_dir):
    os.mkdir(utils.feature_cache_dir)

if not os.path.exists(utils.train_dir):
    os.mkdir(utils.train_dir)

if not os.path.exists(utils.valid_dir):
    os.mkdir(utils.valid_dir)

# this one will always re-write features
# because unlike word extractions, those can actually change

train_check = []
valid_check = []
with open('PatRec17_KWS_Data/task/train.txt', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        train_check.append(line)

with open('PatRec17_KWS_Data/task/valid.txt', 'r') as f:
    lines = f.read().splitlines()
    for line in lines:
        valid_check.append(line)

train = {}
valid = {}
for filename in os.listdir("words"):
    if filename.endswith(".png"):
        image = utils.greyMatrix("words/"+str(filename))
        image = utils.cutoff(image)
        image = utils.resizeArray(image)
        features = utils.featuresExtraction(image)

        name = filename.replace(".png", "")
        id = name.split("-")
        if id[0] in train_check:
            train[name] = features
        elif id[0] in valid_check:
            valid[name] = features

print(('TrainSize', len(train)))
print(('ValidSize', len(valid)))

utils.cache_features(train, valid)

print("Train and Valid sets cached to " + utils.feature_cache_dir)