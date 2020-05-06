import os
from PIL import Image, ImageDraw
import numpy as np
from skimage import filters
import svg.path
from svg.path import parse_path
from bs4 import BeautifulSoup
import cv2

## Constants ##

feature_cache_dir = "feature_cache/"
train_dir = "feature_cache/train/"
valid_dir = "feature_cache/valid/"
distance_dir = "feature_cache/distance/"
plots_dir = "plots/"

dwt_feature_count = 5
dwt_image_size = 100
word_simplification_on = True

## Data (pre-)processing and caching ##

def simplify_word(word):
    '''
    Input: a word transcription
    '''
    if not word_simplification_on:
        return word

    # simplify
    sep = "-"

    # step 1: to lower case and replase strong s with the regular one
    warr = list(
        map(lambda x: x if x != "s_s" else "s", map(lambda x: x.lower(), word.split(sep)))
        )
    # step 2: remove punctuation and 's at the end
    while True:
        lidx = len(warr) - 1 # last entry
        if lidx == 0: break # don't touch single-letter words
        (lchr, slchr) = (warr[lidx], warr[lidx-1])

        #doubles go first
        if slchr == "s_qt" and lchr == "s": warr = warr[:-2] # 's

        #singles
        elif warr[lidx] == "s_cm": warr = warr[:-1] # ,
        elif warr[lidx] == "s_pt": warr = warr[:-1] # .
        elif warr[lidx] == "s_qt": warr = warr[:-1] # '
        elif warr[lidx] == "s_qo": warr = warr[:-1] # :
        elif warr[lidx] == "s_sq": warr = warr[:-1] # ;
        elif warr[lidx] == "s_mi": warr = warr[:-1] # -

        else: break
    #end while

    return sep.join(warr)

def convertTrainSet():
    '''
    Output: produce a collection of pictures of word in a page of the train set
    '''
    with open('PatRec17_KWS_Data/task/train.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            page_to_words(line)

def convertValidSet():
    '''
    Output: produce a collection of pictures of word in a page of the valid set
    '''
    with open('PatRec17_KWS_Data/task/valid.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            page_to_words(line)

def load_truth():
    '''
    Output: a dictionary where "key" is the id of a picture and "value" is the word represented
    '''
    groundTruth = {}
    with open("PatRec17_KWS_Data/ground-truth/transcription.txt", 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            myLine = line.split()
            groundTruth[myLine[0]] = simplify_word(myLine[1])
    return groundTruth

def page_to_words(file):
    '''
    Input: a file number
    Output: a list of png filename stored

    this function is used by convertTrainSet(); convertValidSet()

    Produced picture are rectangle filled with white pixel when needed
    '''

    svg = 'PatRec17_KWS_Data/ground-truth/locations/'+file+'.svg'
    jpg = 'PatRec17_KWS_Data/images/'+file+'.jpg'

    imgs = []

    paths, ids = get_locations(svg)
    img = cv2.imread(jpg)

    for idx, path in enumerate(paths):
        # cropping code source:
        # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python/48301735
        pts = polygon(path)
        pts = np.array(pts).astype(int)

        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2 = bg+ dst

        cv2.imwrite("words/"+ids[idx]+".png", dst2)
        imgs.append(ids[idx]+".png")

    return imgs

def getKeys(word, dict):
    '''
    Input:a dictionary and a word "value"
    Output: list of all "key" such that dict[key] = word
    '''
    keys = []
    for key, value in dict.items():
        if value == word:
            keys.append(key)
    return keys

def load_keyWords():
    '''
    Output: list of word that should be present in the data set.

    WARNING: We observe certain word to be present only one time in the entire data set.
    Some words like "D-o-c-t-o-r" are not present both in the valid and train set.
    '''
    keyWords = set()
    with open("PatRec17_KWS_Data/task/keywords.txt", 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            keyWords.add(simplify_word(line))
    return keyWords

def cache_features(train, valid):
    '''
    Input: Train and Validation sets as dictionaries of word IDs to 1D feature vector
    Output: void
    '''
    for k,v in train.items():
        np.savetxt("feature_cache/train/" + k + ".txt", v)

    for k,v in valid.items():
        np.savetxt("feature_cache/valid/" + k + ".txt", v)

    return None

def load_cached_features():
    '''
    Output: A tuple with Train and Validation sets as dictionaries of word IDs to 1D feature vector
    '''
    train = {}
    valid = {}

    for filename in os.listdir(train_dir):
        k = filename.replace(".txt", "")
        v = np.loadtxt(train_dir + filename)
        train[k] = v

    for filename in os.listdir(valid_dir):
        k = filename.replace(".txt", "")
        v = np.loadtxt(valid_dir + filename)
        valid[k] = v

    return (train, valid)

def load_train_valid_pages():
    '''
    '''
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

    return (train_check, valid_check)

## Image manipulation ##

def greyMatrix(picturePath):
    '''
    Input: a path of a png
    output: a binary np.array

    Consider using a hardcoded threshold: tr = 128
    Potential issues with otsu when an arbitrary background is added to the original picture
    '''

    # get data
    picture = Image.open(picturePath).convert('L')
    img = np.array(picture)
    # binarization
    tr = filters.threshold_otsu(img)
    img = img > tr
    return img

def find_cut(img):#aux function
    i = 0
    while i < len(img):
        if np.any(~img[i]):
            return i
        i += 1
    return 0

def find_cut_end(img):#aux function
    i = len(img) - 1
    while i >= 0:
        if np.any(~img[i]):
            return i
        i -= 1
    return len(img) - 1

def cutoff(img):
    '''
    Input: boolean numpy array with the picture
    Output: image where rows and columns that are fully white are cut out
            only considers the rows/cols that are at the beginning/end
            i.e. changes the "box" to remove useless info which is literally all zeros
    '''
    img = np.array(img)
    cut_above = find_cut(img)
    cut_below = find_cut_end(img) + 1

    cut_left = find_cut(img.T)
    cut_right = find_cut_end(img.T) + 1

    return img[cut_above:cut_below:1,cut_left:cut_right:1]

def polygon(svg_path):
    '''
    Input: a svg path like: "M150 0 L75 200 L225 200 Z"
    Output: a list of point - point here are a list of two float
    '''
    polygon = []
    path = parse_path(svg_path)
    for line in path:
        if type(line) is svg.path.Line:
            polygon.append([line.start.real, line.start.imag])
        elif type(line) is svg.path.Close:
            polygon.append([line.start.real, line.start.imag])
        elif type(line) is svg.path.Move:
            polygon.append([line.start.real, line.start.imag])
        else:
            print("Alert")
            print(type(line))
    return polygon

def get_locations(svg):
    '''
    Input: a svg file path describing contour of words
    Output:
        - paths: a list of svg path
        - ids: the corresponding list of ids
    '''
    svg=open(svg,'r').read()
    soup = BeautifulSoup(svg, 'lxml')
    locations = soup.find_all('path')
    paths = []
    ids = []
    for loc in locations:
        paths.append(loc['d'])
        ids.append(loc['id'])
    return paths, ids

def showNpArray(array):
    image = Image.fromarray(array)
    image.show()

def resizeArray(image):
    image = Image.fromarray(image)
    image = image.resize((100, 100))
    img = np.array(image)
    return img

def show(example):
    '''
    Input: name of a word from either train or valid data
    Displays the processes picture of the word
    Mostly for debug purposes
    Output: void
    '''
    image = Image.open("words/"+str(example) +".png")
    # gray scale image
    image = greyMatrix("words/"+str(example)+".png")
    image = cutoff(image)
    # resized image
    image = resizeArray(image)
    showNpArray(image)

    return None

## DWT helpers ##

def featuresExtraction(image):
    '''
    Input: 100x100 binary np.array
    Ouput: computed features
    '''
    features = []
    # width 1px, offset 1px
    for columns in image.T:
        # size in theory should always be 100
        size = np.prod(columns.shape)
        #previous pixel value; 1 is white, 0 is black; bounding area is white!
        pValue = 1
        #countour
        LC = 0
        UC = 0
        #color ratio
        BP = 0
        WP = 0
        #transition
        BWT = 0

        for index, pixel in np.ndenumerate(columns):
            if pixel == 1:
                #count white pixel
                WP += 1
                if pValue == 0:
                    #we have black to white transition
                    BWT += 1
            elif pixel == 0:
                # count black pixel
                BP += 1
                # last black pixel is the lower countour
                LC = index[0]
                if UC == 0:
                    #first black pixel is the upper countour
                    UC = index[0]
                if pValue == 1:
                    #we have white to black transition
                    BWT += 1
            else:
                print('pixel error' + str((index, pixel)))
            pValue = pixel
        features.append(LC)
        features.append(UC)
        features.append(BWT)
        features.append(BP/size)
        features.append(BP/(LC-UC+1))

        dwt_feature_count = 5 # to be updated if features change

    return features
