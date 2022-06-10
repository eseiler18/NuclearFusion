"""
Custom dataset classes for nuclear fusion time series.
"""
import json
import os
import torch
from typing import List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data_exploration import get_distruption_global_info
from src.path import parquet_name, DATA_CLEAN


class NuclearFusionTimeSeriesDataset(Dataset):
    """Dataset of nuclear fusion time series."""
    def __init__(self, data_dir: str, label_dir: str,
                 receptive_field: int, seq_len: int,
                 label_balance: str = 'const',
                 in_memory: bool = False,
                 verbose: bool = True) -> None:

        self.data_dir = data_dir
        self.label_dir = label_dir
        self.nrecep = receptive_field
        self.nseq = seq_len
        self.label_balance = label_balance
        self.in_memory = in_memory
        self.verbose = verbose

        # get list of shot
        self.filenames = self.get_filenames()
        self.shots = self.file2shot()

        # create sequence
        self.shot2seq()

        self.calc_label_weights()

        self.info()

        if self.verbose:
            self.print_info()

    def shot2seq(self):
        self.shotInd = []
        self.startInd = []
        self.stopInd = []
        self.disruptInd = []
        self.disIdShot1 = []
        # self.disIdShot2 = []
        self.shotWithoutPre = []
        self.shotPreInvalid = []
        self.parquet_memory = dict()
        for file, shot in tqdm(zip(self.filenames, self.shots),
                               total=len(self.shots)):
            data = pd.read_parquet(os.path.join(self.data_dir, file))
            valid_shot = 0
            time = data.index.values
            infoDb = get_distruption_global_info(shot)
            if infoDb["tD"].isna().values[0]:
                self.disIdShot1.append(-1000)
                # self.disIdShot2.append(-1000)
                valid_shot = 1
            else:
                tD_start = self.get_time_PreDisr(shot)
                if tD_start == -1000:
                    self.shotWithoutPre.append(shot)
                else:
                    # tD = infoDb["tD"].values[0]
                    # tD_final = self.get_time_final_disr(shot, tD)
                    indTD_Start = np.argmin(np.abs(np.array(time)-tD_start))
                    if indTD_Start >= data.shape[0]-50:
                        valid_shot = 0
                        self.shotPreInvalid.append(shot)
                    else:
                        # indTD_Final=np.argmin(np.abs(np.array(time)-tD_final))
                        self.disIdShot1.append(int(indTD_Start))
                        # self.disIdShot2.append(int(indTD_Final))
                        valid_shot = 1
            if valid_shot:
                if self.in_memory:
                    self.parquet_memory[str(shot)] = data
                start = 0
                N = data.shape[0]
                nb_seq_frac = (N - self.nseq)/float(self.nseq - self.nrecep+1)
                nb_seq = np.ceil(nb_seq_frac)
                Nseq = self.nseq + (nb_seq - 1)*(self.nseq - self.nrecep + 1)
                start += N-Nseq
                for m in range(int(nb_seq)):
                    self.shotInd += [shot]
                    self.startInd += [start + (m*self.nseq - m*self.nrecep+m)]
                    self.stopInd += [start + ((m+1)*self.nseq-m*self.nrecep+m)]
                    self.startInd[-1] = int(self.startInd[-1])
                    self.stopInd[-1] = int(self.stopInd[-1])
                    if np.logical_and(self.startInd[-1] <= self.disIdShot1[-1],
                                      self.stopInd[-1] >= self.disIdShot1[-1]):
                        self.disruptInd += [self.disIdShot1[-1]]
                    else:
                        self.disruptInd += [-1000]

        self.shotInd = np.array(self.shotInd)
        self.startInd = np.array(self.startInd)
        self.stopInd = np.array(self.stopInd)
        self.disruptInd = np.array(self.disruptInd)
        self.disruptedi = self.disruptInd > 0
        self.length = len(self.shotInd)

    def calc_label_weights(self, inds=None):
        """Calculated weights to use in the criterion"""
        # for now, do a constant weight on the disrupted class, to balance the
        # unbalanced set
        # TODO implement increasing weight towards final disruption
        if inds is None:
            inds = np.arange(len(self.shotInd))
        if 'const' in self.label_balance:
            N = np.sum(self.stopInd[inds] - self.startInd[inds])
            disinds = inds[self.disruptedi[inds]]
            Ndisrupt = np.sum(self.stopInd[disinds] - self.disruptInd[disinds])
            Nnondisrupt = N - Ndisrupt
            self.pos_weight = N/Ndisrupt
            self.neg_weight = 0.5*N/Nnondisrupt
        else:
            self.pos_weight = 1
            self.neg_weight = 1

    def file2shot(self):
        shots = []
        for filename in self.filenames:
            shot_nb = ''
            for m in filename:
                if m.isdigit():
                    shot_nb = shot_nb + m
            shots.append(int(shot_nb))
        return shots

    def get_filenames(self) -> List[str]:
        return os.listdir(self.data_dir)

    def get_time_PreDisr(self, shot):
        f = open(os.path.join(self.label_dir, "JET_Train.json"))
        jet_train = json.load(f)
        pres_disr = jet_train['Class']["PreDisr"]['Summary']
        for disr in pres_disr:
            if disr[0] == shot:
                return disr[1]
        return -1000

    def get_time_final_disr(self, shot, tD):
        f = open(os.path.join(self.label_dir, "JET_Train.json"))
        jet_train = json.load(f)
        for i, s in enumerate(jet_train['ML']["List"]):
            if s == shot:
                t_ML_final = jet_train['ML']["Times_first"][i]
        if t_ML_final < tD:
            return t_ML_final
        else:
            return tD

    def read_data(self, index):
        shot = self.shotInd[index]
        start = self.startInd[index]
        stop = self.stopInd[index]
        if self.in_memory:
            X = self.parquet_memory[str(shot)]
        else:
            X = pd.read_parquet(os.path.join(DATA_CLEAN,
                                             parquet_name(shot, clean=True)))
        return X.values[start:stop, :].transpose()

    def __getitem__(self, index):
        X = self.read_data(index)
        # label for clear(=0) or disrupted(=1, or weighted)
        target = np.zeros((X.shape[-1]), dtype=X.dtype)
        w = self.neg_weight*np.ones((X.shape[-1]), dtype=X.dtype)
        if self.disruptedi[index]:
            # TODO: class weighting beyond constant
            target[self.disruptInd[index]-self.startInd[index]:] = 1
            w[self.disruptInd[index]-self.startInd[index]:] = self.pos_weight
        X = torch.from_numpy(X)
        target = torch.from_numpy(target)
        w = torch.from_numpy(w)
        return X.float(), target.float(), w.float(), index

    def __len__(self):
        return self.length

    def info(self):
        # initial nb of shot
        self.nb_shot = len(self.shots)
        # number of shot used
        self.nb_shot_used = len(np.unique(self.shotInd))
        # ratio distrupted / non distrupted shot
        self.ratio_shot = len(np.unique(self.shotInd[self.disruptedi]))
        self.ratio_shot /= len(np.unique(self.shotInd))
        # ratio distrupted / non distrupted sequence
        self.ratio_seq = sum(self.disruptedi)/len(self.disruptedi)
        # ratio distrupted / non distrupted
        dis_dt = self.stopInd[self.disruptedi]-self.disruptInd[self.disruptedi]
        self.ratio_timestep = sum(dis_dt)/(self.length*self.nseq)
        # nb of shot without pre distruption phase
        self.withoutPre = len(self.shotWithoutPre)
        # nb of shot with invalid pre distruption phase
        self.invalidPre = len(self.shotPreInvalid)
        # nb timestep disruptive in disruptive sequence
        a = self.stopInd[self.disruptedi] - self.disruptInd[self.disruptedi]
        self.nb_distruptive_timestep = a

    def print_info(self):
        print(f"On {self.nb_shot} shots at start {self.nb_shot_used} are used")
        print(f"Remove {self.withoutPre} shots because without pre " +
              " (self.shotWithoutPre)")
        print(f"Remove {self.invalidPre} shot because invalid pre " +
              " (self.invalidPre)")
        print(f"ratio distr shot {self.ratio_shot}")
        print(f"ratio distr seq {self.ratio_seq}")
        print(f"ratio distr ratio_timestep {self.ratio_timestep}")
        print("self.nb_distruptive_timestep to see nb of dist timestep by seq")
