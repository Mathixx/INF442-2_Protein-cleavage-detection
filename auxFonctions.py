class AminoAcid:
    # Dictionary to store amino acid properties
    properties = {
        'A': {'name': 'Alanine', 'label': 1, 'code' : 'A'},
        'C': {'name': 'Cysteine', 'label': 2, 'code' : 'C'},
        'D': {'name': 'Aspartic Acid', 'label': 3, 'code' : 'D'},
        'E': {'name': 'Glutamic Acid', 'label': 4, 'code' : 'E'},
        'F': {'name': 'Phenylalanine', 'label': 5, 'code' : 'F'},
        'G': {'name': 'Glycine', 'label': 6, 'code' : 'G'},
        'H': {'name': 'Histidine', 'label': 7, 'code' : 'H'},
        'I': {'name': 'Isoleucine', 'label': 8, 'code' : 'I'},
        'K': {'name': 'Lysine', 'label': 9, 'code' : 'K'},
        'L': {'name': 'Leucine', 'label': 10, 'code' : 'L'},
        'M': {'name': 'Methionine', 'label': 11, 'code' : 'M'},
        'N': {'name': 'Asparagine', 'label': 12, 'code' : 'N'},
        'P': {'name': 'Proline', 'label': 13, 'code' : 'P'},
        'Q': {'name': 'Glutamine', 'label': 14, 'code' : 'Q'},
        'R': {'name': 'Arginine', 'label': 15, 'code' : 'R'},
        'S': {'name': 'Serine', 'label': 16, 'code' : 'S'},
        'T': {'name': 'Threonine', 'label': 17, 'code' : 'T'},
        'V': {'name': 'Valine', 'label': 18, 'code' : 'V'},
        'W': {'name': 'Tryptophan', 'label': 19, 'code' : 'W'},
        'Y': {'name': 'Tyrosine', 'label': 20, 'code' : 'Y'}
    }

    def __init__(self, aa):
        """
        Initialize the AminoAcid object with a single letter code.
        """
        if aa in self.properties:
            self.code = aa
            self.name = self.properties[aa]['name']
            self.label = self.properties[aa]['label']
        else:
            print('Invalid amino acid code', aa)
            raise ValueError('Invalid amino acid code', aa)

    def __str__(self):
        """
        Print the name of the amino acid.
        """
        return self.code