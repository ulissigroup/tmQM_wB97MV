from ase.io import Trajectory
from ase.io import read, write
from ase.units import Hartree
import numpy as np
from tqdm import tqdm, trange
import sys
sys.path.append('~/ocp')
from ocpmodels.datasets.lmdb_dataset import LmdbDataset

#read file containing the tmqm data
file = read('../all_data/tmQM/tmQM_centered.extxyz', index=slice(None), format='extxyz')

#read the predictions npz from the model
dataset = LmdbDataset({"src":"../training_data/tmQM/80-test.lmdb"})
predictions = np.load('../predictions/tmQM/80-gemnet.npz')

def append_predictions(predictions, test):
    print("Appending predictions")
    pred_dict = {}
    for i in trange(len(predictions['ids'])):
        #if int(predictions['ids'][i]) in pred_dict: print(f"id included twice, id = {int(predictions['ids'][i])}")
        pred_dict[int(predictions['ids'][i])] = predictions['energy'][i]
    if len(pred_dict) != len(test): print("Predictions and Test data do not have the same number of elements")
    new_dataset = []
    for j in trange(len(test)):
        data = test[j]
        if data.sid not in pred_dict:
            print(f'{data.sid} not in predictions.')
            continue
        data.predicted = pred_dict[data.sid]
        data.residual = (data.y_relaxed - data.predicted) * Hartree #residuals in eV
        new_dataset.append(data)
    return new_dataset

pred_dataset = append_predictions(predictions, dataset)

new_CSDs_list = []
for data in tqdm(pred_dataset):
    new_CSDs_list.append(data.CSD_code) 
new_CSDs = set(new_CSDs_list)
if len(new_CSDs_list) != len(new_CSDs): print('Repeats in prediction dataset.')

#define a function to attach the tmqm data to the new dataset
def combine_datasets(new_data, old_data):
    print('Combining datasets:')
    updated_dict = {}
    for i in trange(len(old_data)):
        updated_dict[old_data[i].CSD_code] = {"predicted_target":old_data[i].predicted, "residual":old_data[i].residual, "target": old_data[i].y_relaxed}
    new_dataset = []
    num_omitted = 0
    for j in trange(len(new_data)):
        data = new_data[j]
        CSD = data.info['CSD_code']
        if CSD not in new_CSDs:
            num_omitted += 1
            continue
        old_info = updated_dict.get(CSD)
        data.info['predicted_target'] = old_info['predicted_target']
        data.info['residual'] = old_info['residual']
        data.info['abs_residual'] = np.abs(old_info['residual'])
        data.info['target'] = old_info['target']
        del data.info['Stoichiometry']
        del data.info['|']
        new_dataset.append(data)
    print(num_omitted)
    return new_dataset

new_dataset = combine_datasets(file, pred_dataset)

print('Writing files')
write('../predictions/tmQM/80-gemnet_evpredictions.traj', new_dataset, format='traj')