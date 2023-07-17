# Using Models to Predict

In order to use the models to predict, one needs to create an LMDB of their data, which can be done from an Atoms object in a very similar way as was shown in `tmQM_rev Preprocessing.md` (specifically the method shown in an `*_lmdb_creation.py` script). Then, one runs a model with a checkpoint, but replaces the test data with the data of interest, and then pulls the predictions from that. This is done by creating a new config file (or modifying an existing one), replacing the third `src` under the `dataset` header with one's data of interest, and then running the following command:

`python main.py --mode predict --config-yml /home/jovyan/tmQM_rev/configs/.../config.yml --checkpoint /home/jovyan/tmQM_rev/trained_checkpoints/.../checkpoint.pt`

Where the checkpoint is whatever checkpoint one wants to use from the `trained_checkpoints` directory. New training data could be used in a similar way by replacing the first `src`, and validation data by replacing the second.

These predictions will be reference corrected, since that is what the models are trained on here. One will need to use `revert_reference_correction.py` to convert from reference corrected energies to total energies, which are more commonly reported. This does require that no new elements are present in the predicted structures compared to the original dataset, but that is already recommended to avoid extrapolation.

It is noted that, due to different functionals, it may be wise to compare how well the model predicts energy differences, not absolute energies. For example, if one was predicting the energy of five structures, it is expected that the model should be able to predict their energies relative to one another fairly well, but may have an offset from each structure's absolute energy.