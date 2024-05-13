import pandas as pd
import math
import fonctionsSupervisedLearning2 as fsl2
from sklearn.model_selection import GridSearchCV
import time
# import thundersvm as tsvm

import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib 

import fonctionskernel as fk
from fonctionskernel import p,q




def train_and_evaluate(X_train, bool_train, X_test, bool_test, C, kernel_function):
    """
    Train SVM, evaluate with ROC curve and return the model and its AUC score.
    """
    # model = BaggingClassifier(svm.SVC(C=C, kernel=kernel_function, probability=True, class_weight='balanced'), n_jobs=-1)
    model = svm.SVC(C=C, kernel = kernel_function, probability=True)
    model.fit(X_train, bool_train)
    
    probabilities = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(bool_test, probabilities)
    roc_auc = auc(fpr, tpr)
    
    return model, roc_auc




def run_svm_analysis():
    """
    Run SVM analysis for different configurations, save the best classifier and ROC curves.
    """
    data = pd.read_csv("data/df.csv")
    data = fsl2.convert_df_to_vectors2(data)

    #select 4 first rows of data
    # data = data.head(100)
    
    n = p + q
    # print(data)


    X_train, X_test, bool_train, bool_test = fk.test_train_split_random_pos_proba(data,n)
       # Define parameter grid
    param_grid = {
        'C': [1],  # Example C values
        'kernel': [fk.ProbabilisticKernel],  # Example kernels
        # 'kernel' : [fk.RBF_kernelBLOSUM, fk.RBF_kernelPAM]
    }

    # Setup the SVM classifier with GridSearchCV
    model = svm.SVC(probability=True)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs = -1, return_train_score=True)

    # Start timing and fit GridSearchCV
    start = time.time()
    grid_search.fit(X_train, bool_train)
    end = time.time()

    # Best model evaluation
    print("Best parameters found:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    probabilities = best_model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(bool_test, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic {best_model.kernel} {best_model.C}')

    plt.legend(loc="lower right")
    plt.savefig('data/ROC_Curve_accuracy_rbf.png')
    plt.close()

    # Save the best model
    joblib.dump(best_model, 'data/models/best_svm_rbf.pkl')
    print(f"Best Model Saved with AUC: {roc_auc}, Time to Run: {end - start}s")
    with open("data/accuracy.txt", "w") as f:
        f.write(f"Best Model Saved with AUC: {roc_auc}, Time to Run: {end - start}s")

    # Save GridSearchCV results to a text file
    with open("data/GridSearchCV_results_rbf.txt", "w") as f:
        f.write("Best parameters found: {}\n".format(grid_search.best_params_))
        f.write("GridSearchCV results:\n")
        for i, params in enumerate(grid_search.cv_results_['params']):
            f.write("Configuration {}: {}\n".format(i+1, params))
            f.write("Mean train score: {}\n".format(grid_search.cv_results_['mean_train_score'][i]))
            f.write("Mean test score: {}\n".format(grid_search.cv_results_['mean_test_score'][i]))
            f.write("score time: {}\n".format(grid_search.cv_results_['mean_score_time'][i]))
            f.write("\n")
        f.write("best params accuracy on X_test :" + str(grid_search.score(X_test, bool_test)))



if __name__ == "__main__":
    run_svm_analysis()





