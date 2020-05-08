## How to use

`plots_keywords/` contains all plots for all validation set members for words that are found in `keywords.txt` and have at least one match in training set

The following guide is how to reproduce the result, incl. how to obtain full validation set pictures. Running time was measured on a powerful pc, but inside a VM with linux

#### Step 0 - Install the required libraries
Make sure that you have pip installed and up-to-date
    
    pip install --user -r requirements.txt

#### Step 1 - Preprocessing and feature extraction
Extract words, features and caches both into folders

Running Time : ~1m40s

    python3 cache_all.py

#### Step 2 - DWT
Compute dissimilarity using DWT and cached computed values

Running Time : ~6m30s

    python3 compute.py
    
Compute only for words in keywords.txt

Running Time : ~45s

    python3 compute.py -kw-only
    
#### Step 3 - P/R curves
Compute a P/R curve, AP and plot them on a picture, the result is stored in `plots/` folder

Running Time : ~1m20s

    python3 plotall.py

Compute only for words in keywords.txt

Running Time : ~15s

    python3 plotall.py -kw-only
