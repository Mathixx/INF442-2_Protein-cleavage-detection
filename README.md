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

## Usage


