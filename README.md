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

This Python script provides functionality to predict protein cleavage sites using pre-trained SVM models. It supports multiple kernel types including RBF, BLOSUM, scalar, and probabilistic models. Users can input either a sequence string directly or a file path to a dataset containing multiple sequences. The script can be executed from the command line, taking the kernel type and either the sequence or the path to the dataset as arguments. Based on the kernel specified, the script loads the corresponding model, adjusts parameters p and q for the cleavage prediction, and processes the input to output predicted cleavage sites, highlighting them directly in the sequence output. Results include accuracy, average number of predictions, and average distance to real cleavage sites, which are written to a results file.

To run the script, use the syntax: python predict_cleavage.py <kernel> <sequence or file path>. This enables the prediction of cleavage sites from provided sequences, with detailed logging of the results in a designated text file.


