#from a dataset, reference corrects by creating a matrix of #structures x #unique elements and linearly regressing to guess contribution by individual atoms.  

import sys
sys.path.append("~/ocp/")
from ocpmodels.datasets.lmdb_dataset import LmdbDataset
import lmdb
import pickle
from tqdm import tqdm, trange
import numpy as np

#read dataset you want to reference correct
dataset = LmdbDataset({"src": "../all_data/tmQM_rev/tmqm_rev_elec_e.lmdb"})

#function to find what elements are present in the dataset
def elem_present(dataset):
    elem_list = []
    print("Calculating elements present in dataset")
    for structure in tqdm(dataset):
        for elem in structure.atomic_numbers:
            if int(elem) not in elem_list:
                elem_list.append(int(elem))
    elem_list.sort()
    return elem_list

#function to create the coefficient matrix
#row for each structure, column for each element
#each entry tells the number of times that element is in that structure
def coeff_matrix_func(elem_list, dataset):
    num_structures = len(dataset)
    num_elems = len(elem_list)
    matrix = np.zeros((num_structures, num_elems))
    print("Constructing coefficient matrix")
    for i in trange(len(dataset)):
        elems = dataset[i].atomic_numbers
        for elem in elems:
            for j in range(len(elem_list)):
                if int(elem) == elem_list[j]:
                    matrix[i][j] += 1
    return matrix

#function to solve for the target values attributable to any given element
def target_references(coeff_matrix, dataset):
    sol_matrix = np.zeros(len(dataset))
    for i in range(len(dataset)):
        sol_matrix[i] = dataset[i].y_relaxed
    sol_matrix = sol_matrix.T
    vars = np.linalg.solve(coeff_matrix.T@coeff_matrix, coeff_matrix.T@sol_matrix)
    return vars

#function to subtract off target values attributable to each individual element
def target_correction(coeff_matrix, target_references, dataset):
    print("Correcting targets")
    new_dataset = []
    for i in trange(len(dataset)):
        data = dataset[i]
        reference_target = coeff_matrix[i, :]@target_references
        corrected_target = data.y_relaxed - reference_target
        del data.y_relaxed
        data.y_relaxed = corrected_target
        new_dataset.append(data)
    return new_dataset

#calculate the required matrices
elem_list = elem_present(dataset)
coeff_matrix = coeff_matrix_func(elem_list, dataset)
target_references = target_references(coeff_matrix, dataset)
corrected_dataset = target_correction(coeff_matrix, target_references, dataset)

print("Writing Matrices to an npz:")
np.savez("../reference_correction/tmqm_rev_elec_e_corrections.npz", elem_list=elem_list, coeff_matrix=coeff_matrix, target_references=target_references)
print("Done!")

#matplotlib visualization
print('Making Figures:')

import matplotlib.pyplot as plt

#original figure visualization
original_targets = []
for structure in dataset:
    original_targets.append(structure.y_relaxed)
plt.figure(1)
plt.hist(original_targets, bins=100)
plt.xlabel("Electronic Energy (Hartrees)")
plt.ylabel("Count")
plt.savefig("../visuals/Energy Distributions/tmqm_rev-uncorrected.png", dpi=300)


#modified dataset visualization
modified_targets = []
for structure in corrected_dataset:
    modified_targets.append(structure.y_relaxed)
plt.figure(2)
plt.hist(modified_targets,bins=100)
plt.xlabel("Electronic Energy (Hartrees, Corrected)")
plt.ylabel("Count")
plt.savefig("../visuals/Energy Distributions/tmqm_rev-corrected.png", dpi=300)

#write an LMDB with the reference corrected energies as targets
db = lmdb.open(
    "../all_data/tmQM_rev/tmqm_rev_elec_e_reference_corrected.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

print("Creating LMDB")

for id in trange(len(corrected_dataset)):
    data = corrected_dataset[id]
    txn = db.begin(write=True)
    txn.put(f"{id}".encode("ascii"), pickle.dumps(data,protocol=-1))
    txn.commit()
    db.sync()

db.close()


print("Done!")