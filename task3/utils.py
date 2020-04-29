from PIL import Image, ImageDraw
import numpy as np
from skimage import filters
import svg.path
from svg.path import parse_path
from bs4 import BeautifulSoup
import cv2

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


def load_truth():
    '''
    Output: a dictionary where "key" is the id of a picture and "value" is the word represented
    '''
    groundTruth = {}
    with open("PatRec17_KWS_Data/ground-truth/transcription.txt", 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            myLine = line.split()
            groundTruth[myLine[0]] = myLine[1]
    return groundTruth

def showNpArray(array):
    image = Image.fromarray(array)
    image.show()

def resizeArray(image):
    image = Image.fromarray(image)
    image = image.resize((100, 100))
    img = np.array(image)
    return img

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
    keyWords = []
    with open("PatRec17_KWS_Data/task/keywords.txt", 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            keyWords.append(line)
    return keyWords

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

    return features
