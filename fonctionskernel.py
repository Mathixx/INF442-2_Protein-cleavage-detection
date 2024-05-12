
import pandas as pd
import math
from auxFonctions import AminoAcid
import fonctionsSupervisedLearning2 as fsl2

import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

from Bio.Align import substitution_matrices


# ## Redoing fast the Statistical Study in order to get the s values

# In[3]:


# Read data from a file into a list of entries
with open('data/SIG_13.red', 'r') as file:
    entries = file.read().split('\n   ')


# Define a function to process each entry in the data file
def process_entry(entry):
    lines = entry.split('\n')
    protein_id, primary_structure, annotation = lines
    return {
        'Protein ID': protein_id.split()[1],
        'Primary Structure': primary_structure,
        'Annotation': annotation
    }


# Process each entry
processed_entries = [process_entry(entry) for entry in entries]

# Create a DataFrame
df = pd.DataFrame(processed_entries)
df['Cleavage_Site'] = df['Annotation'].apply(lambda x: x.find('C'))

# Now you can analyze the DataFrame as needed


# In[4]:


# Get the position of the cleavage site
cleavage_site_position = df['Cleavage_Site']


# with have then p = [13, 1] and q = [1, 30]


# In[5]:


# Split the primary structure into a list of amino acids
amino_acid_seq = df['Primary Structure'].apply(lambda x: list(x))

# Create a DataFrame to store, for each primary structure, the neihborhood of the cleavage site
# The neighborhood is defined as the word of length p+q starting p letters before the cleavage site
correct_neighborhood = pd.Series(index=amino_acid_seq.index, dtype=str)
for i, seq in amino_acid_seq.items():
    correct_neighborhood[i] = ''.join(seq[cleavage_site_position[i]-13:cleavage_site_position[i]+2])

# for each amino acid in the sequence, replace it with the corresponding AminoAcid object
amino_acid_seqB = amino_acid_seq
amino_acid_seq = amino_acid_seq.apply(lambda x: [AminoAcid(aa) for aa in x])


# In[6]:


#Parametres de l'étude
p = 13
q = 2

# Create a DataFrame to store the counts of each amino acid at every position relative to the cleavage site
#the cleavage site is between to aminoacids, so cleavage_site_position is the position of the first amino acid after the cleavge site
#So i need to create a dataframe with columns from -p to q without 0
amino_acid_counts = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q))
amino_acid_freqs = pd.DataFrame(0.0, index=AminoAcid.properties.keys(), columns=range(-p, q)) #f(a,i)
amino_acid_pseudo_counts = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q)) #g(a)
amino_acid_s_values = pd.DataFrame(0.0, index=AminoAcid.properties.keys(), columns=range(-p, q)) #s(a,i)


# Count the occurrences of each amino acid at every position relative to the cleavage site

for i, seq in amino_acid_seq.items():
    for j, aa in enumerate(seq):
        position = j - cleavage_site_position[i] #position of the amino acid relative to the cleavage site
        if position in amino_acid_counts.columns:
            amino_acid_counts.loc[aa.code, position] += 1

# Add pseudo-counts to avoid zero counts here pseudocount parameter is 1/len(df)
amino_acid_pseudo_counts = amino_acid_counts + 1

# Print the results
#print("Occurrences of each amino acid at every position relative to the cleavage site:")
#print(amino_acid_pseudo_counts)

# Compute the observed frequency of each amino acid at the relative position (using pseudo-counts)
for i in amino_acid_counts.index:
    for j in amino_acid_counts.columns:
        amino_acid_freqs.loc[i, j] = amino_acid_pseudo_counts.loc[i, j] / len(df)

# Compute the general background frequency of each amino acid
general_background_frequency = amino_acid_freqs.mean(axis=1)

# Compute the s value of each amino acid at every position
for i in amino_acid_counts.index:
    for j in amino_acid_counts.columns:
        amino_acid_s_values.loc[i, j] = math.log(amino_acid_freqs.loc[i, j]) - math.log(general_background_frequency[i])


# In[7]:


# Create a DataFrame to store, for each primary structure, the neihborhood of the cleavage site
# The neighborhood is defined as the word of length p+q starting p letters before the cleavage site
correct_neighborhood = pd.Series()
for i, seq in amino_acid_seqB.items():
    correct_neighborhood[i] = ''.join(seq[cleavage_site_position[i]-p:cleavage_site_position[i]+q])

# Create a DataFrame to store, for each primary structure, a sequence that is not the neighborhood of the cleavage site
incorrect_neighborhood = pd.Series()
decalage  = [1,2,3,4,5, -1,-2,-3,-4, -5]
for i, seq in amino_acid_seqB.items():
    dec = np.random.choice(decalage)
    dec = 0 if cleavage_site_position[i]-13 - dec < 0 else dec
    incorrect_neighborhood[i] = ''.join(seq[cleavage_site_position[i]-p-dec:cleavage_site_position[i]+q-dec])


# In[8]:


# Define the function computing the q-1 score for a given word
def q_minus_1_score(word):
    return sum([amino_acid_s_values.loc[aa, i-p] for i, aa in enumerate(word)])


#A REDEFINIR EN FONCTION DES RESULTATS OBTENUS
threshold = 1.5
# print(threshold)

#A simple thresholding (to be tuned) is then enough to define a simple binary classifier.
def is_cleavage_neighborhood(score):
    return score > threshold

q_minus_1_score('AAAAAAAAAAAAAAA')


# In[9]:


# To obtain the score of the correct neighborhoods, we apply the q-1 score function to each neighborhood
#correct_neighborhood = correct_neighborhood.apply(lambda x: [AminoAcid(aa) for aa in x])
correct_neigboorhood_score = correct_neighborhood.apply(q_minus_1_score)
incorrect_neighborhood_score = incorrect_neighborhood.apply(q_minus_1_score)

#print("Score of the correct neighborhoods:")
#print(correct_neigboorhood_score)
#print("\n")



false_negatives = correct_neigboorhood_score[correct_neigboorhood_score < threshold].count()


false_positives = incorrect_neighborhood_score[incorrect_neighborhood_score > threshold].count()






# ## UTILISATION DES RESULTATS POUR CREER LA KERNEL PROBABILISTE

# In[10]:


def est_un_caractere(obj):
    return isinstance(obj, str) and len(obj) == 1

def Phi(x : chr, y : chr, i : int) :
    if not(est_un_caractere(x) and est_un_caractere(y)) :
        raise ValueError("x and y must be single characters")
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : un acide aminé (sous forme de string)
    - y : un acide aminé (sous forme de string)
    - i : un entier compris entre -p et q-1 (inclus)
    ### Returns:
    - La valeur de la fonction Phi_i(x,y)
    '''
    if (x == y) :
        return ( amino_acid_s_values.loc[x, i] + math.log(1 + math.exp(amino_acid_s_values.loc[x, i])) )
    else :
        return ( amino_acid_s_values.loc[x, i] + amino_acid_s_values.loc[y, i] )

def LogKernel(x : str, y : str, p=13, q=2) :
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : une sequence d'acides aminés de taille p+q (sous forme de string)
    - y : une sequence d'acides aminés de taille p+q (sous forme de string)
    ### Returns:
    - La valeur de la fonction LogKernel(x,y)
    '''
    assert (p+q<=len(x) & p+q <= len(y))
    scores = np.array([Phi(x_s, y_s,i) for x_s, y_s, i in zip(x[0:p+q], y[0:p+q],[j for j in range(-p,q)])])
    sum = np.sum(scores)
        
        
    return sum

"""
def ProbalisticKernel(X, Y):
    # Initialize an empty matrix to store the kernel values
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))

    # Calculate the kernel value for each pair of samples
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            x_str = fsl2.vector_to_word(x)
            y_str = fsl2.vector_to_word(y)
            gram_matrix[i, j] = math.exp(LogKernel(x_str, y_str))

    return gram_matrix
"""

def ProbKernel(X_str,Y_str):
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : une sequence d'acides aminés converti en string
    - y : une sequence d'acides aminés converti en string
    ### Returns:
    - La valeur de la fonction Kernel(x,y)
    '''
    return math.exp(LogKernel(X_str, Y_str))

from joblib import Parallel, delayed
def ProbabilisticKernel(X, Y):
    # Initialize an empty matrix to store the kernel values

    # gram_matrix = np.zeros((X.shape[0], Y.shape[0]))

    # # Calculate the kernel value for each pair of samples
    # for i, x in enumerate(X):
    #     for j, y in enumerate(Y):
    #         gram_matrix[i, j] = ProbKernel(x, y)

    # return gram_matrix
    # Precompute all conversions from vectors to words if possible
    X_str = [fsl2.vector_to_word(x) for x in X]
    Y_str = [fsl2.vector_to_word(y) for y in Y]

    # Initialize an empty matrix to store the kernel values
    n_samples_X, n_samples_Y = len(X_str), len(Y_str)
    gram_matrix = np.zeros((n_samples_X, n_samples_Y))

    # Calculate the kernel value for each pair of samples using parallel processing
    def compute_kernel(i, j):
        return ProbKernel(X_str[i], Y_str[j])

    # Using joblib's Parallel and delayed to parallelize the loop
    results = Parallel(n_jobs=-1)(
        delayed(compute_kernel)(i, j) for i in range(n_samples_X) for j in range(n_samples_Y)
    )

    # Fill the gram matrix with results
    for idx, value in enumerate(results):
        i = idx // n_samples_Y
        j = idx % n_samples_Y
        gram_matrix[i, j] = value

    return gram_matrix



pam250 = substitution_matrices.load("PAM250")
blosum62 = substitution_matrices.load("BLOSUM62")







#Fonction indiquant la similarité entre deux sequences d'acides aminés
def SimilarityPAM(x : str, y : str,p=13,q=2) :
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : une sequence d'acides aminés de taille p+q (sous forme de string)
    - y : une sequence d'acides aminés de taille p+q (sous forme de string)
    ### Returns:
    - La valeur de la fonction Similarity(x,y)
    '''
    scores = np.array([pam250[amino_acid1, amino_acid2] for amino_acid1, amino_acid2 in zip(x[0:p+q], y[0:p+q])])
    sum = np.sum(scores)
    return sum

def SimilarityBLOSUM(x : str, y : str,p=13,q=2) :
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : une sequence d'acides aminés de taille p+q (sous forme de string)
    - y : une sequence d'acides aminés de taille p+q (sous forme de string)
    ### Returns:
    - La valeur de la fonction Similarity(x,y)
    '''
    scores = np.array([blosum62[amino_acid1, amino_acid2] for amino_acid1, amino_acid2 in zip(x[0:p+q], y[0:p+q])])
    sum = np.sum(scores)
    return sum


 
    

def RBF_similarity(x : str, y : str, sigma = 1, SUBSTITUTION_MATRIX = "PAM") :
    '''
    Fonction servant de base a la Kernel de similarité
    ### Parameters:
    - x : une sequence d'acides aminés de taille p+q (sous forme de string)
    - y : une sequence d'acides aminés de taille p+q (sous forme de string)
    - sigma : un reel positif
    ### Returns:
    - La valeur de la fonction Similarity(x,y)
    '''
    if (SUBSTITUTION_MATRIX == "PAM") :
        Similarity = SimilarityPAM
    elif (SUBSTITUTION_MATRIX == "BLOSUM") :
        Similarity = SimilarityBLOSUM
    norm = Similarity(x, x) + Similarity(y, y) - 2 * Similarity(x, y)
    return math.exp(-norm / (2 * sigma**2))

def RBF_kernelPAM(X, Y) :
    '''
    Fonction servant de base a la Kernel RBF (Radial Basis Function) par similarité
    ### Parameters:
    - x : une sequence d'acides aminés converti au prealable en vecteur de taille (p+q)*26 composé de 0 et 1
    - y : une sequence d'acides aminés converti au prealable en vecteur de taille (p+q)*26 composé de 0 et 1
    ### Returns:
    - La valeur de la fonction Kernel(x,y)
    '''
    # Initialize an empty matrix to store the kernel values
    Xb = X
    Yb = Y
    if Xb.ndim == 1:
        Xb = np.array([Xb])
    if Yb.ndim == 1:
        Yb = np.array([Yb])
    gram_matrix = np.zeros((Xb.shape[0], Yb.shape[0]))

    # Calculate the kernel value for each pair of samples
    for i, x in enumerate(Xb):
        X_str = fsl2.vector_to_word(x)
        for j, y in enumerate(Yb):
            Y_str = fsl2.vector_to_word(y)
            gram_matrix[i, j] = RBF_similarity(X_str, Y_str, sigma = 0.5, SUBSTITUTION_MATRIX = "PAM") 

    return gram_matrix

def RBF_kernelBLOSUM(X, Y) :
    '''
    Fonction servant de base a la Kernel RBF (Radial Basis Function) par similarité
    ### Parameters:
    - x : une sequence d'acides aminés converti au prealable en vecteur de taille (p+q)*26 composé de 0 et 1
    - y : une sequence d'acides aminés converti au prealable en vecteur de taille (p+q)*26 composé de 0 et 1
    ### Returns:
    - La valeur de la fonction Kernel(x,y)
    '''
    # Initialize an empty matrix to store the kernel values
    Xb = X
    Yb = Y
    if Xb.ndim == 1:
        Xb = np.array([Xb])
    if Yb.ndim == 1:
        Yb = np.array([Yb])
    gram_matrix = np.zeros((Xb.shape[0], Yb.shape[0]))

    # Calculate the kernel value for each pair of samples
    for i, x in enumerate(Xb):
        X_str = fsl2.vector_to_word(x)
        for j, y in enumerate(Yb):
            Y_str = fsl2.vector_to_word(y)
            gram_matrix[i, j] = RBF_similarity(X_str, Y_str, sigma = 5, SUBSTITUTION_MATRIX = "BLOSUM") 

    return gram_matrix


# ## Traitement des données


def extract_random_subseq(row, n:int, nb_letters:int=26):
    '''
    Extract a random subsequence of length n from the primary structure and the annotation
    ### Parameters:
    - row: a row of the dataframe
    - n: the length of the subsequence
    - nb_letters: the number of letters in the alphabet
    ### Returns:
    - a pandas series containing the subsequence of the primary structure
    -There should be as much valid sequences as invalid sequences
    the subsequence of the annotation, the subsequence of the primary structure as a vector and 
    the position of the cleavage site in the subsequence
    '''
    bool_cleavage = False
    random_double = np.random.random()
    if random_double > 0.5:
        bool_cleavage = True

    if bool_cleavage:
        start_index = row['Cleavage_Site'] - 13
        end_index = start_index + n  #n = 13 + 2 = 15

        neighborhood_check = 1  # Define wheter the sequence if the right neighborhood of the cleavage site
    else :
        max_start_index = max(0, len(row['Primary Structure']) - n)  # Calculate the maximum possible start index
        if max_start_index == 0:
            start_index = 0  # if chain is too short, start at the beginning
        else:
            start_index = np.random.randint(0, max_start_index)  # Randomly select a start index
        end_index = start_index + n  # Calculer l'indice de fin

        neighborhood_check = 1 if (row['Cleavage_Site'] - start_index == 13) else 0  # Define wheter the sequence if the right neighborhood of the cleavage site
        
    return pd.Series([row['Primary Structure'][start_index:end_index], row['P_Structure_vector'][start_index*nb_letters:end_index*nb_letters], neighborhood_check], index=['Primary Structure', 'P_Structure_vector', 'Neighborhood_bool'])


# In[16]:


def test_train_split_random_pos_proba(df, n ,test_size=0.2, random_state=42):
    '''
    Split the data into training and testing sets
    ### Parameters:
    - df: the dataframe containing the data
    - n: the length of the subsequence
    - test_size: the proportion of the data to include in the test split
    - random_state: the seed for the random number generator
    ### Returns:
    - X_train: the training set
    - X_test: the testing set
    - pos_train: the position of the cleavage site in the training set
    - pos_test: the position of the cleavage site in the testing set
    '''
    df_random = df.apply(extract_random_subseq, axis=1, n=n)
    X = np.array(df_random['P_Structure_vector'].tolist())
    
    y = np.array(df_random['Neighborhood_bool'].tolist())   

    X_train, X_test, bool_train, bool_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    """
    test_size=0.2: This argument specifies the proportion of the dataset to include in the test split. 
    In this case, 20% of the data will be used for testing, and the remaining 80% will be used for training

    random_state=42: This argument sets the seed for the random number generator that shuffles the data before splitting. 
    Setting a specific seed (like 42 in this case) ensures that the output is reproducible, i.e., 
    you'll get the same train/test split each time you run the code.
    """

    return X_train, X_test, bool_train, bool_test



