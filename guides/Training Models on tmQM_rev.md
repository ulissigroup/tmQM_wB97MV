# Training Models
Once the preprocessing shown in the `tmQM_rev Preprocessing.md` file is finished, training models from OCP is very simple and can be done from the command line.

If one has already installed OCP and performed the necessary environment setup, training OCP models from the command line is as simple as navigating to the OCP folder (default name `ocp`), and running the following command:

`python main.py --mode train --config-yml ../tmQM_rev/configs/.../config.yml`

Where one replaces the path to the YML file with whatever path is relevant, in this case, the YML files used can be found under the `configs` folder. It is also often helpful to include a `--identifier` flag to identify the run. The model will then train, and the files that are most relevant are the checkpoint of the trained model, which can be found under `ocp/checkpoints`, and the results, which are written as an `npz` file under `ocp/results`. It is noted that the results file, named `is2re_predictions`, is the prediction of the model on the test set using the weights from the last epoch. To get predictions using the best epoch (lowest validation MAE), use the command:

`python main.py --mode predict --config-yml ../tmQM_rev/configs/.../config.yml --checkpoint ../ocp/checkpoints/.../best_checkpoint.pt`

Where the checkpoint is `best_checkpoint.pt` from the `ocp/checkpoints` directory, and the config is exactly the same as used for training.

The predictions on the test set for each model trained in this work (using the best-performing checkpoint) can be found in the `predictions` folder.

For a better understanding of the models, configs, and commands, see the OCP repository. However, some basic information is given here, as an interpretation of the config files. We use the `energy` trainer since we are only predicting energies, and not forces. Under the `dataset` header, there are three `src` lines, corresponding to the training, validation, and test set, in that order. We do normalize the data, so `normalize_labels` is True, and the mean and standard deviation are provided. The `logger` is Weights and Biases, but one could use tensorboard if that is preferred. The `task` section is fairly standard, and sets the dataset type, task, etc. The `model` section gives which GNN should be used, as well as some of the hyperparameters. `use_pbc` and `regress_forces` should be False since these systems are not periodic and there are no forces. However, `otf_graph` should be True since edges are computed on-the-fly during training. The parameters under `optim` can also be adjusted to affect optimizer performance.

Instead of training models from scratch, one can also just use the pretrained model for predictions or do finetuning.

To use a pretrained model from this work, one used the checkpoints found in the `trained_checkpoints` folder. To use these to predict the energies of structures in the test set, use:

`python main.py --mode predict --config-yml ../tmQM_rev/configs/.../config.yml --checkpoint ../tmQM_rev/trained_checkpoints/.../checkpoint.pt`

Where the checkpoint is whichever pretrained model one wants to use. To do a finetuning run, one would use the same command as above, except replace the config with whatever config they wanted to use (e.g. lowering the learning rate and training on different training/validation data), and set the mode to train. More sophisticated finetuning runs could also be done. To do predictions on other datasets, see the `Using Models to Predict.md` file. Starting from the models trained on OC20 is also possible, by using the publicly available checkpoints on the OCP repo.

In order to analyze the performance of the models, the MAE, EwT, and parity were assessed, using the `parity_plots.py` script. This takes in a dataset with targets and the models' predictions. It returns a parity plot and various information on performance, like MAE.

Visualizations were also performed using chemiscope. In order to make the ASE files required for the ASAP package, the `tmqm_rev_asap_residual_append.py` script was used, which attaches the predictions made by a model to the original ASE Atoms objects, which will then be transferred into the `.json.gz` file needed for https://chemiscope.org/ using the ASAP software package (used from the command line). The files used for the visualizations in this work can be found in the `chemiscope_files` directory. An analogous `tmqm_asap_residual_append.py` script exists for tmQM.