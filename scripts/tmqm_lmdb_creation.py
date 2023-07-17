from ase.io import read
from tqdm import tqdm
import csv
import lmdb
import pickle
import sys
sys.path.append("~/ocp/")
from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

#read extxyz file for data
raw_data = read('../all_data/tmQM/tmQM_centered.extxyz', index=slice(None), format='extxyz')

#convert Atoms to Data object

a2g = AtomsToGraphs(max_neigh=50, radius=6, r_energy=False, r_forces=False, r_distances=False, r_edges=False,r_fixed=False)

#convert atoms objects into graphs
data_objects = []
for idx, system in enumerate(raw_data):
    data = a2g.convert(system)
    data.sid = idx
    data_objects.append(data)

#read csv file of targets, append onto data structure
with open('~/tmqm/data/tmQM_y.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            data_objects[line_count-1].CSD_code = row[0]
            data_objects[line_count-1].Electronic_E = float(row[1])
            data_objects[line_count-1].Dispersion_E = float(row[2])
            data_objects[line_count-1].Dipole_M = float(row[3])
            data_objects[line_count-1].Metal_q = float(row[4])
            data_objects[line_count-1].HL_Gap = float(row[5])
            data_objects[line_count-1].HOMO_Energy = float(row[6])
            data_objects[line_count-1].LUMO_Energy = float(row[7])
            data_objects[line_count-1].Polarizability = float(row[8])
            data_objects[line_count-1].y_relaxed = float(row[1]) #need target to be named y_relaxed
            line_count += 1

'''
print('Filtering dataset:')
dataset = []

for idx, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    q = data.q
    if q == 0:
        dataset.append(data)
data_objects = dataset
'''

#write LMDB
db = lmdb.open(
    "../all_data/tmQM/tmqm_elec_e.lmdb",
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
