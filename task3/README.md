## How to use

`plots_keywords/` contains all plots for all validation set members for words that are found in `keywords.txt` and have at least one match in training set

The following guide is how to reproduce the result, incl. how to obtain full validation set pictures. Running time was measured on a powerful pc, but inside a VM with linux

#### Step 0 - install the required libraries
  pip install --user -r requirements.txt

#### Step 1 - preprocessing and feature extraction

  python3 cache_all.py
* extracts words, extracts features and caches both into folders
* Running time: ~1m40s

#### Step 2 - DWT

  python3 compute.py
* computes dissimilarity using DWT and caches computed values
* Running time: ~6m30s

  python3 compute.py -kw-only
* same but only does it for words in keywords.txt
* Running time: ~45s

#### Step 3 - P/R curves

  python3 plotall.py
* computes a P/R curve, AP and plots them on a picture, the result is stored in `plots/` folder
* Running time: ~1m20s

  python3 plotall.py -kw-only
* same but only does it for words in keywords.txt
* Running time: ~15s
