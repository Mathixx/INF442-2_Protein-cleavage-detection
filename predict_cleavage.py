import pandas as pd
import fonctionsSupervisedLearning1 as fsl1


# import thundersvm as tsvm

import numpy as np
import sys


import joblib 

model_paths = {
    'rbf': 'data/models/best_svm_rbf.pkl',
    'blosum': 'data/models/svm_blosum.pkl',
    'poly': 'data/models/best_svm_model_accuracy_scalar.pkl',
    'probabilistic': 'data/models/best_svm_proba.pkl'}

models_p = {
    'rbf': 13,
    'blosum': 13,
    'poly': 13,
    'probabilistic': 13
}

models_q = {
    'rbf': 2,
    'blosum': 2,
    'poly': 2,
    'probabilistic': 2
}




def predict_sequence(X, model, p=13, q = 2, nb_letters = 26):
    '''
    X is the string sequence
    model is the model to predict the cleavage site
    '''
    n = p+q
    # convert string to vector
    X_vect = fsl1.word_to_vector(X)

    # find the probable cleavage site
    predicted_sites =  fsl1.find_cleavage2(X_vect, model, p, q)
    # print in bold the corresponding letters
    print("Sequence: ", end="")
    for i in range(len(X)):
        if i in predicted_sites:
            print("\033[1m" + X[i] + "\033[0m", end="")
        else:
            print(X[i], end="")
    print()
    print("Predicted sites: ", predicted_sites)


def predict_file(df, model, p=13, q=2, nb_letters = 26):
    X = df['P_Structure_vector']
    pos = df['Cleavage_Site']
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
    
    with open("data/results/results_predict_cleavage.txt", "a") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Average number of predictions: {avg_pred}\n")
        f.write(f"Average distance to the real cleavage site: {avg_dist}\n")
        f.write("model:" + kernel + "\n")
        f.write("data: " + arg + "\n")
        f.write("\n")


if __name__ == "__main__":
     # The first argument is the script name, ignore it
    # The second argument will be the kernel name
    # The third argument will be the sequence to predict
    if len(sys.argv) > 2:
        kernel = sys.argv[1]
        model = joblib.load(model_paths[kernel])
        p = models_p[kernel]
        q = models_q[kernel]
        arg = sys.argv[2]

    try:
        # Try to open the file
        with open(arg, 'r') as file:
            # If the file opens successfully, read the sequence from the file
            entries = file.read().split('\n   ')
            processed_entries = [fsl1.process_entry(entry) for entry in entries]

        # Create a DataFrame
            df = pd.DataFrame(processed_entries)
            df['Cleavage_Site'] = df['Annotation'].apply(lambda x: x.find('C'))
            df['P_Structure_vector'] = df['Primary Structure'].apply(fsl1.word_to_vector)
            predict_file(df, model, p, q)

    except FileNotFoundError:
        # If the file does not open, assume the argument is a sequence string
        X = arg
        predict_sequence(X, model, p, q)
else:
    print("syntax: python predict_cleavage.py <kernel> <sequence or file path>")
    sys.exit(1)
