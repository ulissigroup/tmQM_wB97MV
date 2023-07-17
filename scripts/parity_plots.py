#Given the original dataset, the test dataset, and the predictions, makes a parity plot

import sys
sys.path.append("~/ocp/")
from ocpmodels.datasets.lmdb_dataset import LmdbDataset
import lmdb
import pickle
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             median_absolute_error)
from ase.units import Hartree

#test dataset
test = LmdbDataset({"src": "../training_data/tmQM_neutral/80-test.lmdb"})

#predictions file from model
predictions = np.load("../predictions/tmQM_neutral/80-gemnet.npz")

#function to append predictions to original structures
def append_predictions(predictions, test):
    print("Appending predictions")
    pred_dict = {}
    for i in trange(len(predictions['ids'])):
        pred_dict[int(predictions['ids'][i])] = predictions['energy'][i]
    if len(pred_dict) != len(test): print("Predictions and Test data do not have the same number of elements")
    new_dataset = []
    for j in trange(len(test)):
        data = test[j]
        data.predicted_target = pred_dict[data.sid] 
        new_dataset.append(data)
    return new_dataset

pred_dataset = append_predictions(predictions, test)

#create parity plot
uncorrected_targets = []
uncorrected_predictions = []
AE_per_atom = []

for structure in pred_dataset:
    uncorrected_targets.append(structure.y_relaxed)
    uncorrected_predictions.append(structure.predicted_target)
    
    residual = structure.y_relaxed - structure.predicted_target
    abs_residual = np.abs(residual)
    num_atoms = structure.natoms
    AE_per_atom.append(abs_residual / num_atoms)
    

x = np.array(uncorrected_targets)
y = np.array(uncorrected_predictions)

lims = [np.min(x)*1.2, np.max(x)*1.2]
grid = sns.jointplot(x=x, y=y, kind='hex', bins='log', xlim=lims, ylim=lims)
ax = grid.ax_joint
_ = ax.set_xlim(lims)
_ = ax.set_ylim(lims)
_ = ax.plot(lims, lims, '--')
_ = ax.set_xlabel('Targets (Electronic Energy, Hartrees)')
_ = ax.set_ylabel('Predictions (Electronic Energy, Hartrees)')

cbar_ax = grid.fig.add_axes([1.05, 0.2, .05, .4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)

mae_peratom = np.mean(np.array(AE_per_atom))
mae = mean_absolute_error(x, y)
rmse = np.sqrt(mean_squared_error(x, y))
mdae = median_absolute_error(x, y)

marpd = np.abs(2 * (y-x) /
               (np.abs(y) + np.abs(x))
               ).mean() * 100
r2 = r2_score(x, y)
corr = np.corrcoef(x, y)[0, 1]

print(f'The MAE in hartrees is: {mae}.')
print(f'The MAE in meV is: {mae*Hartree * 1000}.')

print(f'The MAE in Hartree/atom is: {mae_peratom}.')
print(f'The MAE in meV/atom is: {mae_peratom*Hartree * 1000}.')

threshold = 0.043 #eV
num_within_threshold = 0

for i in range(len(x)):
    hartree_diff = np.abs(y[i] - x[i])
    ev_diff = hartree_diff * Hartree
    if ev_diff < threshold:
        num_within_threshold += 1
        
ewt = num_within_threshold / len(x) * 100

print(f'The EwT is {ewt}%.')

hartrees_to_ev = Hartree
mae_peratom *= hartrees_to_ev * 1000
mdae *= hartrees_to_ev
mae *= hartrees_to_ev
rmse *= hartrees_to_ev

'''
text = ('  MAE = %.2f eV\n' % mae + 
        '  RMSE = %.2f eV\n' % rmse + 
        '  R2 = %.2f\n' %r2 +
        '  EwT = %i%%\n' %ewt)
'''

text = ('  MAE = %.2f meV/atom \n' % mae_peratom + 
        '  R2 = %.2f\n' %r2 +
        '  EwT = %.2f%%\n' %ewt)


_ = ax.text(x=lims[0], y=lims[1], s=text,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=12)

plt.savefig('../visuals/tmQM_neutral/80-gemnet-testset.png', bbox_inches='tight', dpi=300)