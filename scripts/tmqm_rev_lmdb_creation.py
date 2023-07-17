from ase.io import Trajectory, read
from tqdm import tqdm
import lmdb
import pickle

import sys
sys.path.append("~/ocp/")

from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

#read traj file for data
raw_data = Trajectory('../all_data/tmQM_rev/tmQM_rev.traj', mode='r')

#convert Atoms to Data object

a2g = AtomsToGraphs(max_neigh=50, radius=6, r_energy=False, r_forces=False, r_distances=False, r_edges=False,r_fixed=False)

#convert Atoms objects to graphs, transfer information
data_objects = []
csds = []
for idx, system in tqdm(enumerate(raw_data), total=len(raw_data)):
    data = a2g.convert(system)
    data.sid = idx
    data.CSD_code = system.info['CSD_code']
    data.q = system.info['q']
    data.spin = system.info['spin']
    data.energy = system.info['energy']
    data.formula = system.info['formula']
    data.y_relaxed = system.info['energy'] #target must be named y_relaxed
    if data.CSD_code not in csds:
        data_objects.append(data)
        csds.append(data.CSD_code)
    

print('Filtering dataset:')
dataset = []

'''
for idx, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    q = data.q
    if q == 0:
        dataset.append(data)
data_objects = dataset
'''

db = lmdb.open(
    "../all_data/tmQM_rev/tmqm_rev_elec_e.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

idx = 0
for sid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    #assign sid
    data.sid = idx

    txn = db.begin(write=True)
    txn.put(f"{sid}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()
    db.sync()

    idx += 1


db.close()

print("Done!")
