import csv
import operator
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit

import warnings
warnings.filterwarnings('always')

#Global variables
X = [] #features
y = [] #labels

optimal_kernel = None
optimal_gamma = None
optimal_C = None

def load_data(path, nb_samples):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        matrix = np.array(data, dtype=int)
        features_set = matrix[:, 1:]
        labels_set = matrix[:, 0]

    #Separate feature and labels
    training_length = nb_samples
    global X,y
    X = features_set[:training_length, :]
    y = labels_set[:training_length]

def parameters_optimization():
    #Separate data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #Configure parameters
    kernel_params = ['poly', 'rbf']
    C_params = [0.1, 1, 10, 100, 1000]
    gamma_params = [10**-3, 10**-2, 10**-1, 1]
    param_grid = {'kernel': kernel_params, 'C': C_params , 'gamma': gamma_params}

    grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Average accuracy during cross-validation for all investigated kernels")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in sorted(zip(means, stds, grid.cv_results_['params']),reverse=True, key=operator.itemgetter(0)):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    #Set optimal values
    global optimal_kernel, optimal_C, optimal_gamma
    optimal_kernel = grid.best_params_['kernel']
    optimal_gamma = grid.best_params_['gamma']
    optimal_C = grid.best_params_['C']

def classify_optimized():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    classifier = SVC(kernel=optimal_kernel, gamma=optimal_gamma, C=optimal_C)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    print("Classification with following parameters : ")
    print("Kernel : " + optimal_kernel)
    print("Gamma : " + str(optimal_gamma))
    print("C : " + str(optimal_C))
    print(metrics.classification_report(y_test, predictions))

    confusion_m = metrics.confusion_matrix(y_test, predictions)
    print("Confusion matrix : ")
    print(confusion_m)

    print("Accuracy={}".format(metrics.accuracy_score(y_test, predictions)))

def svm(path, nb_samples):
    load_data(path, nb_samples)
    parameters_optimization()
    classify_optimized()


svm('../dataset/mnist_test.csv', 1000)


