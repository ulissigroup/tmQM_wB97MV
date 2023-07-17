#Given the original dataset, the test dataset, and the predictions, returns an LMDB with the test structures and their predicted targets as an attribute

import sys
sys.path.append("~/ocp/")
from ocpmodels.datasets.lmdb_dataset import LmdbDataset
import lmdb
import pickle
from tqdm import trange
import numpy as np

#test dataset (reference corrected)
test = SinglePointLmdbDataset({"src": "../training_data/tmQM_rev/80-test.lmdb"})

#predictions file from model
predictions = np.load("../predictions/tmQM_rev/80-gemnet.npz")

#npz for linreg corrections
corrections = np.load("../reference_correction/tmqm_rev_elec_e_corrections.npz")

#function to make the coefficient matrix for the test set
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

#function to append predictions to original structures
def append_predictions(predictions, test):
    print("Appending predictions")
    pred_dict = {}
    for i in trange(len(predictions['ids'])):
        if int(predictions['ids'][i]) in pred_dict: print(f"id included twice, id = {int(predictions['ids'][i])}")
        pred_dict[int(predictions['ids'][i])] = predictions['energy'][i]
    if len(pred_dict) != len(test): print("Predictions and Test data do not have the same number of elements")
    new_dataset = []
    for j in trange(len(test)):
        data = test[j]
        data.predicted_target = pred_dict[data.sid] 
        new_dataset.append(data)
    return new_dataset

#function to add target values attributable to each individual element
def target_reversion(coeff_matrix, target_references, dataset):
    print("Converting predictions and targets")
    new_dataset = []
    for i in trange(len(dataset)):
        data = dataset[i]
        reference_target = coeff_matrix[i, :]@target_references
        '''
        #use only if reverting predictions on the original test set (SIDs from the original coeff matrix)
        sid = data.sid
        reference_target = coeff_matrix[sid, :]@target_references
        '''
        reverted_prediction = data.predicted_target + reference_target
        del data.predicted_target
        data.predicted_target = reverted_prediction
        new_dataset.append(data)
    return new_dataset

#for writing the dataset with adjusted predictions to an LMDB
elem_list = corrections['elem_list']
#coeff_matrix = corrections['coeff_matrix'] #use only if reverting predictions on the original test set
coeff_matrix = coeff_matrix_func(elem_list, test)
target_references = corrections['target_references']
pred_dataset = append_predictions(predictions, test)
corrected_predictions = target_reversion(coeff_matrix, target_references, pred_dataset)


#write LMDB with the reverted predictions
db = lmdb.open(
    "../predictions/tmQM_rev/80-gemnet.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

print("Creating LMDB")

for id in trange(len(corrected_predictions)):
    data = corrected_predictions[id]
    txn = db.begin(write=True)
    txn.put(f"{id}".encode("ascii"), pickle.dumps(data,protocol=-1))
    txn.commit()
    db.sync()

db.close()

print("Done!")
