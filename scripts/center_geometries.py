import numpy as np

from tqdm import tqdm

import ase
from ase.io import read, write, Trajectory
from ase.formula import Formula

# metals present in tmQM
metal_symbols = {'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 
                 'Zn': 30, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 
                 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'La': 57, 'Hf': 72, 'Ta': 73, 
                 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80}

metal_symbols_list = list(metal_symbols.keys())

# non-metals present in tmQM
non_metals_symbols = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 
                       'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 
                       'Br': 35, 'I': 53, 'As': 33, 'Se': 34}

non_metals_symbols_list = list(non_metals_symbols.keys())


# Read file 
file = read('../all_data/tmQM_combined_notcentered.extxyz', index=slice(None), format='extxyz')

print('Number of Structures: ', len(file))

# Function to center complex at the origin
def center_complex(geom, metal):
    atoms = geom.copy()
    for atom in atoms:
        if atom.symbol == metal:
            metal_pos = np.array(atom.position)
            atoms.translate(-1*metal_pos)
    return atoms

# Function to get the metal label
def get_metal_label(geom):
    atoms = geom.copy()
    geom_formula =atoms.get_chemical_formula()
    W = Formula(geom_formula).count()
    elements = W.keys()
    metal = [el for el in elements if el not in non_metals_symbols][0]
    return str(metal)

# Loop over each geometry and store new geom 
centered = []

for geom in tqdm(file):
    metal_symbol = get_metal_label(geom)
    geom_centered = center_complex(geom, metal_symbol)
    centered.append(geom_centered)

# Export new file
write('../all_data/tmQM/tmQM_centered.extxyz', centered, format='extxyz')

print('Done!')
