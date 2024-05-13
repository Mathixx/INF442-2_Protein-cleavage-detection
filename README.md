# Protein Cleavage Site Prediction

This project aims to predict the position of cleavage sites in protein sequences. The sequences of amino acids, starting from the N-terminal side, are used to represent proteins. Each amino acid is denoted by a letter code, with most uppercase letters used in this encoding. 

## Project Overview

The project begins with a simple statistical model to establish a reference. We then aim to improve accuracy using specific kernel functions for Support Vector Machines (SVMs).

## Data

The data used in this project consists of sequences of amino acids. An example of a protein sequence is shown below, where the cleavage site is marked as the bond between the two underlined AR amino acids:

## Installation


To use the different models we fitted and selected :
    load the desired model using : joblib -> loaded_model = joblib.load('best_svm_model_accuracy_PROBA.pkl')

    each of these models have been trained on different lengths of cleavage site neighborhoods
    Here are the following lengths :
        - BLOSUM Model -> p = 13, q = 2 , n = 15
        - Probabilistic Model -> p = 9, q = 2, n = 11
        - Model using the scalar product -> p = 13, q = 2, n = 15

## Predict Cleavage Sites with SVM Models
predict_cleavage.py
This Python script predicts protein cleavage sites using pre-trained SVM models with various kernels (RBF, BLOSUM, scalar, probabilistic). Input can be a sequence string or a file path to a dataset with multiple sequences.

    Models and Paths:
        rbf: data/models/best_svm_rbf.pkl
        blosum: data/models/best_svm_model_accuracy_BLOSUM.pkl
        scalar: data/models/best_svm_model_accuracy_scalar.pkl
        probabilistic: data/models/best_svm_model_accuracy_PROBA.pkl

    Parameters:
        p=13, q=2 for all models

    Usage:
        Run: python predict_cleavage.py <kernel> <sequence or file path>
        Example: python predict_cleavage.py rbf "MAGTMAASSAAGLAGLGLAAG"

    Output:
        Displays the sequence with predicted cleavage sites in bold.
        Writes accuracy, average number of predictions, and average distance to real cleavage sites to data/results/results_predict_cleavage.txt.


## other files
We used other files and notebooks to tune our parameters.


