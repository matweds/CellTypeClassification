'''
Program to format the morphologie images into structured directories

Authors: Isak Bengtson, Mattias Wedin
'''

import pandas as pd
import numpy as np
import sys

MORPHOLOGIE_DATA_FILE_MOUSE = "data/mouse.json"

if __name__ == "__main__":
    print("dope")
    
    # Read json file
    f = open(MORPHOLOGIE_DATA_FILE_MOUSE)
    print(data_path)
    data = json.load(f)
    
    # The chosen cell types
    cell_types = ['pyramidal', 'granule', 'medium spiny', 'sensory receptor',
                    'bipolar',
                    'ganglion', 'amacrine','dopaminergic',
                    'Motoneuron', 'Somatostatin (SOM)-positive',
                    'Parvalbumin (PV)-positive OR basket OR Fast-spiking',
                    'Neuropeptide Y (NPY)-positive', 'Martinotti','Purkinje','thalamocortical',
                    'stellate', 'prion protein (PrP) promoter-positive',
                    'Vasoactive Intestinal Peptide (VIP)-positive',
                    'Steroid Hormone Receptor Ad4BP-positive',
                    'callosal-projecting','periglomerular','mitral',
                    'Cajal-Retzius', 'Serotonin receptor type 3A(5HT3)-positive',
                    'tufted', 'neurogliaform', 'Plasma Retinol-Binding Protein-Positive',
                    'bitufted', 'tyrosine-hydroxylase-positive', 'Golgi' ,'translocating',
                    'low-threshold calcium spiking', 'Chandelier', 'Thick-tufted',
                    'shrub cell', 'corticothalamic', 'horizontally elongated',
                    'Neuronal Nitric Oxide Synthase (nNOS)-positive',
                    'deep projecting cell', 'single bouquet', 'Midbrain-projecting']
    
    bad_cell_types = ['Glia', 'microglia']




