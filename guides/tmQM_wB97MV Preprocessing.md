# tmQM_wB97MV Preprocessing

In order to prepare the datasets for ML model training, several steps were followed, detailed below. This file will extensively reference the scripts contained in the `scripts` folder.

Since this work bases its structures on those in tmQM, the data from tmQM was used. However, the original tmQM data is not needed to replicate any of the tmQM_wB97MV results in this work, except for running the original filtering, described in the `tmQM Filtering` notebook, for running the `center_geometries` script, and for replicating models trained on tmQM. For convenience, we include all relevant data from tmQM in this repository.

The first preprocessing step is centering the structures in tmQM. We did this using the `center_geometries.py` script. This takes the `extxyz` file containing non-centered structures, taken from the tmQM GitHub repository by the method described in the `tmQM Filtering.ipynb` notebook, and then shifts all of the structures such that the one metal atom present is at the origin. It then writes a new `extxyz` containing all of the centered structures. These structures were the inputs to the GNNs trained for both tmQM and tmQM_wB97MV. The user of this repository should not have to use `center_geometries.py` to replicate any results since the structures provided are already centered. However, `center_geometries.py` can be used to center new structures that one might want to predict on. Note that the script is only designed to work with tmQM, but similar scripts can be used for other coordination complexes.

For the remaining steps, the OCP repository must be installed, which can be found at https://github.com/Open-Catalyst-Project/ocp.

We first must take the ASE Atoms objects and convert them into graphs stored in LMDBs in order to train OCP models on them.

We do this with `tmqm_wB97MV_lmdb_creation.py`. This takes the original, centered file (`all_data/tmQM_wB97MV/tmQM_wB97MV.traj`), converts the ASE Atoms objects to graphs, appends the necessary information, and writes an LMDB file, which can be used to train ML models. Note that there is a `y_relaxed` attribute assigned to the data objects in the LMDB (set equal to the electronic energy), which is the target name used in OCP models.

There is an analogous `tmqm_lmdb_creation.py` script used to make LMDBs from tmQM, although to use this file one must download the `csv` file on the original tmQM GitHub.

Note that, in both of these scripts, there is a commented out section that can be used for filtering. Specifically, if uncommented, it will only write the neutral structures to the LMDB, instead of the entire dataset. Other filters can also be implemented if desired.

Using the entire dataset, for each subset, we computed the mean and the standard deviation using `np.mean` and `np.std`. These values can be found in the config files included in the `configs` directory (on a given config `yml`, under the `dataset` header, the `target_mean` and `target_std` labels give the computed values for the entire dataset).

We then perform reference correction, which converts the total energies reported in tmQM and tmQM_wB97MV to something more like a formation energy by subtracting off the average energy of each atom in the dataset, computed by linear regression. This was done using the `reference_correction.py` script.

What the reference correction script does is take an LMDB, calculate the elements present in the entire dataset, construct a coefficient matrix giving the number of times each element appears in each structure for the dataset, computes the average energy attributable to each element over the entire dataset via linear regression, subtracts those energies for each atom in each structure, and then writes a new LMDB with that reference corrected energy as the target (note that the new LMDB will still have an `energy` (for tmQM_wB97MV) or `Electronic_E` (for tmQM) attribute that is not reference corrected, but the targets, `y_relaxed`, will be reference corrected). This script writes the matrices it uses for these calculations as `npz` files, contained in the `reference_correction` folder. There are three matrices, `elem_list`, which gives a list of the atomic numbers of the elements present in a dataset, `coeff_matrix`, a matrix that is (number of structures) by (number of elements), where each entry corresponds to the number of times that element appears in that structure, and `target_references`, which gives the average energy attributable to each element.

Finally, we create training, validation, and testing splits of our data using the `random_split_subsets.py` script. These splits, which are referenced in the configuration files in the `configs` directory, along with the means and standard deviations computed earlier, are all that is needed to train models from OCP.
