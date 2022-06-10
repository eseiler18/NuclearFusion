# Disruption prediction for nuclear fusion with Wavenet on JET dataset 

* [Emilien Seiler](mailto:emilien.seiler@epfl.ch)

Semester project of Emilien Seiler, Master in Computational Science and Engineering at EPFL. 
Collaboration project between Computer Vision Laboratory (CVLab) and the Swiss Plasma Center (SPC) at EPFL.

## Instaling
To setup a miniconda environement
```
conda env create -f <path_to_environement.yml_file>
```

## To train model
```
cd project/script
```
```
python train.py --epoch <nb_epoch> --batch <batch_size> --parquet-dir <path_to_parquet>
```
On lac10 server recommend batch-size <= 12.

Model and log of the training will be save in project/output at the end of the training

## Data
parquet data are in the /tmp/apau file of lac10


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
