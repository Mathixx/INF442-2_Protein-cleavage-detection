import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def process_entry(entry):
    '''Process each entry in the data file and return a dictionary with the protein ID, primary structure, and annotation.
    ### Parameters:
    - entry (str): The entry to process
    ### Returns:
    - dict: A dictionary with the protein ID, primary structure, and annotation'''
    try:
        lines = entry.split('\n')
        protein_id, primary_structure, annotation = lines
        return {
            'Protein ID': protein_id.split()[1],
            'Primary Structure': primary_structure,
            'Annotation': annotation
        }
    except:
        print(entry)



# Define a mapping from letters to integer codes

le = LabelEncoder()
le.fit(list(map(chr, range(ord('A'), ord('Z')+1))))


def word_to_vector(word):
    '''Convert a word into a vector
    :param word: a string
    :return: a numpy array
    '''
    vec = np.zeros(26 * len(word))
    for i, char in enumerate(word):
        vec[i * 26 + le.transform([char])[0]] = 1
    return vec

def vector_to_word(vec):
    # Define a function to decode a vector into a word
    word = ''
    for i in range(0, len(vec), 26):
        word += le.inverse_transform([np.argmax(vec[i:i+26])])[0]
    return word

def convert_df_to_vectors2(df):
    '''
    Convert the dataframe to a format that can be used for training a classifier
    add a column 'Annotation_pos' that contains the position of the cleavage site
    add a column 'P_Structure_vector' that contains the primary structure as a vector
    '''
    df_exploitable = df.copy()
    df_exploitable['Cleavage_Site'] = df_exploitable['Annotation'].apply(lambda x: x.find('C'))
    df_exploitable['P_Structure_vector'] = df_exploitable['Primary Structure'].apply(word_to_vector)
    return df_exploitable


def extract_random_subsequence2(row, n = 15, p = 13,  nb_letters:int=26):
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
        start_index = row['Cleavage_Site'] - p
        end_index = start_index + n  #n = 13 + 2 = 15

        neighborhood_check = 1  # Define wheter the sequence if the right neighborhood of the cleavage site
    else :
        max_start_index = max(0, len(row['Primary Structure']) - n)  # Calculate the maximum possible start index
        if max_start_index == 0:
            start_index = 0  # if chain is too short, start at the beginning
        else:
            start_index = np.random.randint(0, max_start_index)  # Randomly select a start index
        end_index = start_index + n  # Calculer l'indice de fin

        neighborhood_check = 1 if (row['Cleavage_Site'] - start_index == p) else 0  # Define wheter the sequence if the right neighborhood of the cleavage site
    
    return pd.Series([row['Primary Structure'][start_index:end_index], row['P_Structure_vector'][start_index*nb_letters:end_index*nb_letters], neighborhood_check], index=['Primary Structure', 'P_Structure_vector', 'Neighborhood_bool'])


def test_train_split_random_pos2(df, n , p,test_size=0.2, random_state=42):
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
    
    df_random = df.apply(extract_random_subsequence2, axis=1, n=n, p = p)
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


def find_cleavage2(X, svm_model_neighbor,n,  nb_letters = 26):
    '''
    find the position of the cleavage site in the primary structure using two SVM models
    /!\ the models must be trained before using this function with the same n and nb_letters as the ones used in this function
    ### Parameters:
    - X: the primary structure as a vector
    - svm_model_neighbor: the SVM model that predicts if the subsequence of size n is in the neighborhood of the cleavage site (position 13)
    - threshold: the threshold for the confidence of the prediction
    ### Returns:
    - the position of the cleavage site if the prediction is confident enough, otherwise Nan
    '''
    proba_position = []
    containing = False
    for i in range(0, len(X)- n*nb_letters, nb_letters):
        test_sub = X[i:i + n*nb_letters]
        
        if svm_model_neighbor.predict([test_sub]):
            containing = True
            position = i//nb_letters+13
            return True
    return False

def is_neighborhood2(X, svm_model_neighbor,  nb_letters = 26):
    
    X = word_to_vector(X)
    if svm_model_neighbor.predict([X]):
        return True
    return False


def create_model2(n, p, df_exploitable, kernel_neighbor, C_nei, random_state=42, nb_letters = 26):
    '''
    Create a model that predicts the position of the cleavage site in a primary structure
    ### Parameters:
    - n: the length of the subsequence
    - nb_letters: the number of different letters in the alphabet
    - df_exploitable: the dataframe containing the data
    - random_state: the seed for the random number generator
    - kernel_nei: the kernel used for the SVM model that predicts if the subsequence is a neighborhood of the cleavage site
    - C_nei: the regularization parameter for the SVM model that predicts if the subsequence is a neighborhood of the cleavage site
    ### Returns:
    - svm_model_in: the SVM model that predicts if the subsequence is a neighborhood of the cleavage site
    - accuracy_in: the accuracy of the model that predicts if the subsequence is a neighborhood of the cleavage site

    '''
    X_train, X_test, bool_train, bool_test = test_train_split_random_pos2(df_exploitable, n,p, random_state=random_state)
    #in_train = ~np.isnan(bool_train)
    #in_test = ~np.isnan(bool_test)
    svm_model_neighbor = svm.SVC(kernel=kernel_neighbor, C=C_nei, random_state=random_state)
    svm_model_neighbor.fit(X_train, bool_train)
    in_pred = svm_model_neighbor.predict(X_test)

    accuracy_nei = accuracy_score(bool_test, in_pred)

    return svm_model_neighbor, accuracy_nei
    

def test_models2(n, df_exploitable, svm_model_neighbor, random_state=42, nb_letters = 26):
    '''
    Test the model that predicts the position of the cleavage site in a primary structure
    ### Parameters:
    - n: the length of the subsequence
    - nb_letters: the number of different letters in the alphabet
    - df_exploitable: the dataframe containing the data
    - random_state: the seed for the random number generator
    - svm_model_in: the SVM model that predicts if the subsequence contains the cleavage site
    - svm_model_pos: the SVM model that predicts the position of the cleavage site in the subsequence
    ### Returns:
    - accuracy_in: the accuracy of the model that predicts if the subsequence contains the cleavage site
    - accuracy_pos: the accuracy of the model that predicts the position of the cleavage site in the subsequence

    '''
    X_train, X_test, bool_train, bool_test = test_train_split_random_pos2(df_exploitable, n, random_state=random_state)
    nei_pred = svm_model_neighbor.predict(X_test)
    accuracy_nei = accuracy_score(bool_test, nei_pred)

    return accuracy_nei