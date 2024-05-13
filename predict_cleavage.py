import pandas as pd



# import thundersvm as tsvm

import numpy as np



import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import joblib 


def predict(X, model, p=13, q = 2, nb_letters = 26):
    '''
    X is the strinc sequence
    model is the model to predict the cleavage site
    '''
    n = p+q
    positions = []
    for i in range(0, len(X)- n*nb_letters, nb_letters):
        test_sub = X[i:i + n*nb_letters]
        
        if model.predict(np.array([test_sub])):
            position = p+i//nb_letters
            positions.append(position)
    return positions
