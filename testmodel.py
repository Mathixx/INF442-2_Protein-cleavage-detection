import pandas as pd
import math
from auxFonctions import AminoAcid
import fonctionsSupervisedLearning2 as fsl2
import fonctionsSupervisedLearning1 as fsl1


# import thundersvm as tsvm

import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import joblib 
import fonctionskernel as fk


# open model
model = joblib.load("data/models/best_svm_model_accuracy2.pkl")

# open data
data = pd.read_csv("data/df.csv")
data = fsl2.convert_df_to_vectors2(data).head(100)


X = data['P_Structure_vector']
# X = np.array(X)

# X = X.reshape(1,-1)

pos = data['Cleavage_Site']

# predict
def main():
    # find the cleavage for the whole dataset
    predictions = [fsl1.find_cleavage2(x, model) for x in X]
    
    # number of correct predictions
    correct = 0
    for i in range(len(predictions)):
        if pos[i] in predictions[i]:
            correct += 1
    
    #average accuracy
    accuracy = correct/len(predictions)

    #average number of predictions:
    avg_pred = sum([len(x) for x in predictions])/len(predictions)

    #average distance to the real cleavage site, if pred is not empty
    flat_list = [abs(p - pos[i]) for i, x in enumerate(predictions) if x for p in x]
    avg_dist = sum(flat_list) / len(flat_list) if flat_list else 0
    print("Average accuracy: ", accuracy)
    print("Average number of predictions: ", avg_pred)
    print("Average distance to the real cleavage site: ", avg_dist)





if __name__ == "__main__":
    main()
    

