# Disruption prediction for nuclear fusion with Wavenet on JET dataset 

Semester project of [Emilien Seiler](mailto:emilien.seiler@epfl.ch), Master in Computational Science and Engineering at EPFL. 
Collaboration project between Computer Vision Laboratory (CVLab) and the Swiss Plasma Center (SPC) at EPFL.

## Installing
To setup a miniconda environement
```
conda env create -f <path_to_environement.yml_file>
```

## Structure

This is the structure of the repository:

- `data`: 
  - `label`: folder with shot information concerning distruption
  - `shot_info.xlsx`: table with information concerning missing channel in shots
- `models`:
  - `WaveNet.py`: WaveNet implementation
  - `info model.txt` : hyperparameter of the model
- `notebook`: 
  - `plot_training.ipynb`: notebook to plot training result
- `output`:
  - `bestwavenet-150chan-900epoch-01062022.pt`: best model
  - `log-150chan-900epoch-01062022.pickle`: log of best model
- `script`:
  - `train.py`: script to train 
  - `eval.py`: script to eval
- `src`:
  - `data_exploration.py`: functions for data exploration
  - `path.py`: path management
  - `data_preprocessing.py`: functions for data preprocessing
  - `dataset.py`: dataset pytorch for JET
  - `metric.py`: metrics
  - `plot_utils.py`: functions to plot
- `environment.yml`: environment.yml file with all dependencies to setup a conda environnement
- `Report_EmilienSeiler_SemesterProject_MasterCSE.pdf`; report of the project


## Data
Parquet data are provide on the SPC lac 10 cluster in the `/tmp/apau/` folder  
Dataset1 with 150 channels in `/parquet150chan`  
Dataset2 with 372 channels in `/parquet372chan`

## Train
```
python train.py --epoch <nb_epoch> --batch <batch_size> --parquet-dir <path_to_parquet> --input-channels <channel 150 or 372>
```
To train on a pretrained model use:
- `--prtrain-model`: str, name of the model file
- `--prtrain-log`: str, name of the log file
- `--prtrain-dir`: str, directory of the pretrained file

To train on a Wavenet with other hyperparameters:
- `--kernel-size`: int
- `--stack-size`: int
- `--layer-size`: int
- `--nrecept`: depand of the three hyperparameter look in the report Eq. 4
- `--dropout`: float

Other parameter
- `--in-memory`: boolean, to keep all data in memory if you have enougth RAM
- `--lr`: float, initial learning rate
- `--validation`: boolean, split data in train and test set
- `--split`: float, split ratio

On lac10 cluster recommend batch-size < 12.  
Model and log of the training will be save in `project/output` at the end of the training
