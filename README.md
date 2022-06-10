# Nucelar fusion Jet DataSet project [Author: Emilien Seiler Master EPFL 2020]

* [Emilien Seiler](mailto:emilien.seiler@epfl.ch)

Code for the master semester project of Emilien Seiler at EPFL. Collaboration project between Computer Vision Laboratory (CVLab) and the Swiss Plasma Center (SPC) at EPFL

## Documentation
to setup a miniconda environement
```
conda env create -f <path_to_environement.yml_file>
```

### To train model
```
cd project/script
```
```
python train.py --epoch <nb_epoch> --batch <batch_size> --parquet-dir <path_to_parquet>
```
On lac10 server recommend batch-size <= 12.

Model and log of the training will be save in project/output at the end of the training

### Data
parquet data are in the /tmp/apau file of lac10

