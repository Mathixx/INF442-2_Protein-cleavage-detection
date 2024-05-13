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
import time


# open model
model = joblib.load("data/models/best_svm_rbf.pkl")

# open data
data = pd.read_csv("data/df.csv")
data = fsl2.convert_df_to_vectors2(data).head(100)


X = data['P_Structure_vector']
# X = np.array(X)

# X = X.reshape(1,-1)

pos = data['Cleavage_Site']

# predict
def main():
    start = time.time()
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
    end = time.time()
    
    with open("data/results_sliding_rbf.txt", "a") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Average number of predictions: {avg_pred}\n")
        f.write(f"Average distance to the real cleavage site: {avg_dist}\n")
        f.write("model: best_svm_model_matrix\n")
        f.write("data: df.csv\n")
        f.write(f"Time: {end-start}\n")
        f.write("\n")





if __name__ == "__main__":
    main()
    

