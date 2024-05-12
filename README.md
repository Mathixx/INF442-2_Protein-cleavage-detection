# 442-2-Protein-cleavage
Projet 2 de INF442

We consider the problem of predicting the position of cleavage site in proteins, relying on patterns learned from a training set. Here a protein is known as its primary sequence, i.e. the sequence of its amino acids, which is given starting from the N-terminal side. Each amino acid is denoted by a letter code and, as there are 20 standard amino acids, plus some peculiarities, most of upper case letters are used in this encoding. Let A = {A, . . . , Z} be this alphabet.
For instance, the following sequence is the beginning of a protein, where the cleavage site is marked as the bond between the two underlined AR amino acids :

             MASKATLLLAFTLLFATCIARHQQRQQQQNQCQLQNIEALEPIEVIQAEA...

Then, for the purpose of this project, one will simply work on such sequences of letters. One has now to program some learning algorithms, to tune them, and to evaluate their performance. A simple statistical model, will be used first to establish a reference. One will then try to improve accuracy with some specific kernel functions for Support Vector Machines.


Ce qu'il reste a faire
tester le modele simple sur les neighboorhood de la base de données et verifier que l'algo les decrit bien comme tels
On peut considerer une base de données plus grande en mixant les 4 dataframes (avec concatenate (pas dur)) -> meme pas besoin, le fichier sig_13 est deja la concatenation des 3 fichiers
Attention peut etre un traitement necessaire comme pour le fichier 1

On peut changer la taille considérée du neighboorhood et prendre la taille maximale que l'on a a disposition



A FAIRE : 
-Coder subsitution Matrix (2 formes) et adapter le code de la Kernel RBF -> comment changer le produit d'une RBF
- S'assurer que le produit de base et le produit scalaire simple -> sinon le coder et refit le modele

- 

pip install biopython

from Bio.Align import substitution_matrices

# Load the PAM250 matrix
pam250 = substitution_matrices.load("PAM250")

# Access scores directly
score_AR = pam250[('A', 'R')]
print("Score for A and R:", score_AR)

score_AA = pam250[('A', 'A')]
print("Score for A and A:", score_AA)


-debuggeer probabilistic kernel
    -2 methodes pour le faire 


# Protein Cleavage Site Prediction

This project aims to predict the position of cleavage sites in protein sequences. The sequences of amino acids, starting from the N-terminal side, are used to represent proteins. Each amino acid is denoted by a letter code, with most uppercase letters used in this encoding. 

## Project Overview

The project begins with a simple statistical model to establish a reference. We then aim to improve accuracy using specific kernel functions for Support Vector Machines (SVMs).

## Data

The data used in this project consists of sequences of amino acids. An example of a protein sequence is shown below, where the cleavage site is marked as the bond between the two underlined AR amino acids:


## Installation

To run the code in this project, you need to have Python installed on your machine. You can then install the required libraries using pip:


## Usage

The main functions in this project are contained in the `fonctionsSupervisedLearning2.py` file. These include functions for creating the SVM model, splitting the data into training and testing sets, and calculating the kernel function.

To use these functions, you can import them into your Python script as follows:

```python
from fonctionsSupervisedLearning2 import create_model2, test_train_split_random_pos2