
import pandas as pd
import math
from auxFonctions import AminoAcid
import numpy as np

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

# Now you can analyze the DataFrame as needed
# Get the position of the cleavage site
cleavage_site_position = df['Annotation'].apply(lambda x: x.find('C'))
#print("Position of the cleavage site:")
#print(cleavage_site_position)
print("Average position of the cleavage site:")
print(cleavage_site_position.mean())
print("\n")

print("The extremum position of the cleavage site:")
print(cleavage_site_position.min())
print(cleavage_site_position.max())

print("The distance from cleavage site to the C-terminal part of the amino-sequence:")
#there is always 30 amino-acids after the cleavage site
print("\n")\

# with have then p = [13, 1] and q = [1, 30]

# Split the primary structure into a list of amino acids
amino_acid_seq = df['Primary Structure'].apply(lambda x: list(x))
amino_acid_seqBis = df['Primary Structure'].apply(lambda x: list(x))


# for each amino acid in the sequence, replace it with the corresponding AminoAcid object
amino_acid_seq = amino_acid_seq.apply(lambda x: [AminoAcid(aa) for aa in x])

#Parametres de l'étude
p_opt, q_opt = 1, 1
fn_min = 1408
fp_min = 1408
for p in range(2, 14):
    for q in range(1, 10):
        # print("p = ", p, "    q = ", q)
        # print("\n")

        # Create a DataFrame to store, for each primary structure, the neihborhood of the cleavage site
        # The neighborhood is defined as the word of length p+q starting p letters before the cleavage site
        correct_neighborhood = pd.Series()
        for i, seq in amino_acid_seqBis.items():
                correct_neighborhood[i] = ''.join(seq[cleavage_site_position[i]-p:cleavage_site_position[i]+q])

        incorrect_neighborhood = pd.Series()
        decalage  = [3,4,5,-3,-4, -5]
        for i, seq in amino_acid_seqBis.items():
            dec = np.random.choice(decalage)
            dec = 0 if cleavage_site_position[i]-13 - dec < 0 else dec
            incorrect_neighborhood[i] = ''.join(seq[cleavage_site_position[i]-p-dec:cleavage_site_position[i]+q-dec])

        

        # Create a DataFrame to store the counts of each amino acid at every position relative to the cleavage site
        #the cleavage site is between to aminoacids, so cleavage_site_position is the position of the first amino acid after the cleavge site
        #So i need to create a dataframe with columns from -p to q without 0
        amino_acid_counts = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q))
        amino_acid_freqs = pd.DataFrame(0.0, index=AminoAcid.properties.keys(), columns=range(-p, q))
        amino_acid_pseudo_counts = pd.DataFrame(0, index=AminoAcid.properties.keys(), columns=range(-p, q))
        amino_acid_s_values = pd.DataFrame(0.0, index=AminoAcid.properties.keys(), columns=range(-p, q))


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

        # Define the function computing the q-1 score for a given word
        def q_minus_1_score(word):
            return sum([amino_acid_s_values.loc[aa, i-p] for i, aa in enumerate(word)])

        # To obtain the score of the correct neighborhoods, we apply the q-1 score function to each neighborhood
        #correct_neighborhood = correct_neighborhood.apply(lambda x: [AminoAcid(aa) for aa in x])
        correct_neigboorhood_score = correct_neighborhood.apply(q_minus_1_score)
        incorrect_neighborhood_score = incorrect_neighborhood.apply(q_minus_1_score)

        correct_mean = correct_neigboorhood_score.mean()
        std_dev = correct_neigboorhood_score.std()
      
        incorrect_mean = incorrect_neighborhood_score.mean()
        std_dev_incorrect = incorrect_neighborhood_score.std()

        threshold_opti = (correct_neigboorhood_score.mean() - std_dev/math.sqrt(0.9) + incorrect_neighborhood_score.mean() + std_dev_incorrect/math.sqrt(0.9))/2


        false_negatives = correct_neigboorhood_score[correct_neigboorhood_score < threshold_opti].count()
        false_positives = incorrect_neighborhood_score[incorrect_neighborhood_score > threshold_opti].count()


        if false_negatives + false_positives < fn_min + fp_min:
            fn_min = false_negatives
            fp_min = false_positives
            p_opt = p
            q_opt = q


        
print("Optimal values of p and q for the deviation:")
print("p = ", p_opt, "    q = ", q_opt)
print("\n")
print("Minimal amount of false_negatives:")
print(fn_min)
print("Minimal amount of false_positives:")
print(fp_min)
        


















#Here I'm saving my results in the result.txt file
def save_results():
    with open('results.txt', 'a') as file:
        file.write("###     RESULTS :#####\n\n")
        file.write("Parameters of the model:\n")
        file.write("Data obtained from the EUKSIG_13.red file\n\n")
        file.write("p = 13    q = 2\n\n")
        file.write("Average position of the cleavage site:\n")
        file.write(str(cleavage_site_position.mean()) + "\n\n")
        file.write("Occurrences of each amino acid at every position relative to the cleavage site:\n")
        amino_acid_counts_str = amino_acid_counts.to_string()
        file.write(amino_acid_counts_str + "\n\n")
        file.write("Observed frequency of each amino acid at the relative position:\n")
        amino_acid_freqs_str = amino_acid_freqs.to_string()
        file.write(amino_acid_freqs_str + "\n\n")
        file.write("General background frequency of each amino acid:\n")
        file.write(str(general_background_frequency) + "\n\n")
        file.write("s value of each amino acid at every position:\n")
        amino_acid_s_values_str = amino_acid_s_values.to_string()
        file.write(amino_acid_s_values_str + "\n\n")
        file.write("END OF RESULTS\n\n\n")
    
#save_results()


