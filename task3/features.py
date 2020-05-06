'''
Make sure to have a "words" folder in your working directory

This Program output a collection of picture of words: xxx-xx-xx.jpg  where the xxx-xx-xx is the id of the word represented the image.
Each of this picture is a portion of an original handwritten scanned page.

The the program transform those words images into black and white (binary) and resize them to 100x100 pictures.
After that it extrat features for each pictures.

The final part of the program find a keyword and print the original version, the black and white, the resized and the features vector
for up to two of its occurence in the train/valid set.

Do not forget to manually close those (up to six) images
'''

import os
import utils
import random
from PIL import Image



#########################
# 1. Create word pictures
#########################
print("STEP 1: Convert page.jpg into word.png")

utils.convertTrainSet()
print("train set done")
utils.convertValidSet()
print("valid set done")

##############
# 2. Load data
##############
print("STEP 2: load data")

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

print("Train and Valid set loaded")

keyWords = utils.load_keyWords()
groundTruth = utils.load_truth()

print("ground truth loaded")


################
# 3. One example
################
print("STEP 3: Find a keyword in the data set")
word = random.choice(keyWords)
# word = "i-m-m-e-d-i-a-t-e-l-y"
examples = utils.getKeys(word, groundTruth)
print(word)
print(examples)

i = 0
for example in examples:
    if i <2:
        # stored word image
        image = Image.open("words/"+str(example) +".png")
        image.show()
        # gray scale image
        image = utils.greyMatrix("words/"+str(example)+".png")
        utils.showNpArray(image)
        # resized image
        image = utils.resizeArray(image)
        utils.showNpArray(image)
        # features
        id = example.split("-")
        if id[0] in train_check:
            print("in the train set")
            print(train[example])
        elif id[0] in valid_check:
            print("in the valid set")
            print(valid[example])
