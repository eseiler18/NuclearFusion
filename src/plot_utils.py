"""plot utils
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
import torch
import torch.nn as nn
from src.data_exploration import get_tD
from src.path import DATA_CLEAN, LABEL_PATH, parquet_name


def plot_feature(feature: pd.Series) -> None:
    """plot feature evolution vs time

    Args:
        feature (pd.Series): feature to plot
    """
    fig = plt.figure(1, figsize=(14, 3))
    ax1 = plt.subplot(1)
    plt.plot(feature)
    ax1.set_title(feature.name)
    ax1.set_ylabel("value")
    ax1.set_xlabel("time")
    fig.tight_layout()
    plt.show()


def plot_feature_withDB(feature: pd.Series, Db_info: pd.DataFrame) -> None:
    """plot feature evolution vs time with respect to distruption informations
    RU, RD, FT, TD

    Args:
        feature (pd.Series): feature to plot
        Db_info (pd.DataFrame): distruption informations
        RU, RD, FT, TD, CQ_80_20
    """
    fig = plt.figure(1, figsize=(14, 3))
    ax1 = plt.subplot()
    ax1.plot(feature)
    bottom, top = ax1.get_ylim()
    ax1.add_patch(
        Rectangle((Db_info["FT_1"].values[0], bottom),
                  Db_info["FT_2"].values[0]-Db_info["FT_1"].values[0],
                  top-bottom, facecolor="cyan", label="Flat-Top",
                  edgecolor='b'))
    ax1.add_patch(
        Rectangle((Db_info["RD_1"].values[0], bottom),
                  Db_info["RD_2"].values[0]-Db_info["RD_1"].values[0],
                  top-bottom, facecolor="yellow", label="Ramp-down",
                  edgecolor='y'))
    ax1.add_patch(
        Rectangle((Db_info["RU_1"].values[0], bottom),
                  Db_info["RU_2"].values[0]-Db_info["RU_1"].values[0],
                  top-bottom, facecolor="palegreen", label="Ramp-Up",
                  edgecolor='g'))
    if ~Db_info["tD"].isna().values[0]:
        ax1.plot([Db_info["tD"].values[0], Db_info["tD"].values[0]],
                 [bottom, top], color="red", label="time of distruption")
    ax1.set_title(feature.name + " with distruption phase")
    ax1.set_ylabel("value")
    ax1.set_xlabel("time")
    fig.tight_layout()
    plt.legend()
    plt.show()


def plot_feature_with_globalDB(feature: pd.Series,
                               Db_info: pd.DataFrame) -> None:
    """plot feature evolution vs time with respect to distruption global
    informations

    Args:
        feature (pd.Series): feature to plot
        Db_global_info (pd.DataFrame): distruption gloabal informations
    """
    color = ["silver", "rosybrown", "darksalmon", "sandybrown", "darkkhaki",
             "palegreen", "seagreen", "ligthseagreen", "paleturquoise",
             "darkturquoise", "mediumpurpule", "cyan", "mediumvioletred",
             "palevioletred", "royalblue", "k", "chartreuse", "tan",
             "skyblue", "salmon", "spring", "powderblue"]
    fig = plt.figure(1, figsize=(14, 3))
    ax1 = plt.subplot()
    ax1.plot(feature)
    bottom, top = ax1.get_ylim()
    for i, first in zip(range(1, Db_info.shape[1]-2),
                        Db_info.columns.values[1:Db_info.shape[1]-2]):
        if i % 2 == 1:
            index = first.find("first")
            last = first[0:index] + "last"
            label = first[0:index-1]
            if ~Db_info[first].isna().values[0]:
                if Db_info[first].values[0] != Db_info[last].values[0]:
                    width = Db_info[last].values[0]-Db_info[first].values[0]
                    heigth = top-bottom
                    ax1.add_patch(
                        Rectangle((Db_info[first].values[0], bottom), width,
                                  heigth, color=color[i//2], label=label))
                else:
                    ax1.plot([Db_info[first].values[0],
                              Db_info[first].values[0]],
                             [bottom, top], color=color[i//2], label=label)
    if ~Db_info["tD"].isna().values[0]:
        ax1.plot([Db_info["tD"].values[0], Db_info["tD"].values[0]],
                 [bottom, top], color="red", label="time of distruption")
    ax1.set_title(feature.name + " with global distruption parameter")
    ax1.set_ylabel("value")
    ax1.set_xlabel("time")
    fig.tight_layout()
    plt.legend(loc="upper left")
    plt.show()


def plot_PF(data: pd.DataFrame) -> None:
    """plot peak factor : ECE_Te, BOLO_H_cam, HRTS_Te, HRTS_Ne, Li
    and event that lead to distruption

    Args:
        data (pd.DataFrame): data to plot
    """
    fig, axs = plt.subplots(4, figsize=(10, 10))
    axs[0].plot(data["ECE_PF"], color="darkgreen", label="ECE Te")
    axs[0].plot(data["HRTS_Te"], color="lime", label="HRTS Te")
    axs[0].plot(data["ECM1_PF"], color="greenyellow", label="HRTS Te")
    axs[0].legend()
    axs[0].title.set_text("Temperature")
    axs[1].plot(data["BOLO_HV"], color="purple", label="Bolo H cam")
    axs[1].plot(data["BOLO_V_outer"], color="slateblue",
                label="Bolo H_outer cam")
    axs[1].plot(data["BOLO_XDIV"], color="magenta", label="Bolo H_div cam")
    axs[1].title.set_text("Radiations")
    axs[1].legend()
    axs[2].plot(data["HRTS_Ne"], color="aqua", label="HRTS Ne")
    axs[2].title.set_text("Electron density")
    axs[2].legend()
    axs[3].plot(data["LI"], color="k", label="Li")
    axs[3].title.set_text("Inductance")
    axs[3].legend()
    fig.suptitle('Time evolution of the PFs')
    fig.tight_layout()
    fig.supxlabel("Time")
    fig.supylabel("Peaking Factor")
    plt.show()


def plot_PF_with_globalDB(data: pd.DataFrame, Db_info: pd.DataFrame) -> None:
    """plot peak factor : ECE_Te, BOLO_H_cam, HRTS_Te, HRTS_Ne, Li
    and event that lead to distruption

    Args:
        data (pd.DataFrame): data to plot
    """
    pfs = ["ECE_PF", "HRTS_Te", "ECM1_PF", "BOLO_HV", "BOLO_V_outer",
           "BOLO_XDIV", "HRTS_Ne", "LI"]
    for pf in pfs:
        plot_feature_with_globalDB(data[pf], Db_info)


def plot_loss(train: list, test: list = None):
    """plot train and (optionnal) test loss

    Args:
        train (list): train loss
        test (list, optional): test loss. Defaults to None.
    """
    fig = plt.figure(1, figsize=(14, 3))
    ax1 = plt.subplot()
    ax1.plot(train, color='b', label="train loss")
    if test is not None:
        ax1.plot(test, color='r', label="test loss")
    ax1.set_title("Losses vs epochs")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    fig.tight_layout()
    plt.legend(loc="best")
    plt.show()


def plot_metric(acc: list, f1: list):
    """plot accuracy and f1 score

    Args:
        acc (list): _description_
        f1 (list): _description_
    """
    fig = plt.figure(1, figsize=(14, 3))
    ax1 = plt.subplot()
    ax1.plot(acc, color='b', label="accuracy")
    ax1.plot(f1, color='r', label="f1 score")
    ax1.set_title("Metrics vs epochs on test set")
    ax1.set_ylabel("Metric")
    ax1.set_xlabel("Epoch")
    fig.tight_layout()
    plt.legend(loc="best")
    plt.show()


def plot_conf_matrix(tp: int, tn: int, fp: int, fn: int):
    """plot confusion matrix

    Args:
        tp (int): true positive
        tn (int): true negative
        fp (int): false positive
        fn (int): false negative
    """
    cf_matrix = [[tn, fp], [fn, tp]]
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted class')
    ax.set_ylabel('Actual class')

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()


def plot_training(history: dict, validation: bool):
    """plot losses, metric and confusion matrix of training

    Args:
        history (dict): log of the training (in OUT directory)
        validation (bool): True if validation during training
    """
    train_loss = history["train_loss"]
    if validation:
        test_loss = history["test_loss"]
        acc = history["test_acc"]
        f1 = history["test_f1"]
        conf = history["conf_mat"]
        conf = conf[-1]
        tp = conf[0]
        tn = conf[1]
        fp = conf[2]
        fn = conf[3]
    if validation:
        plot_loss(train_loss, test_loss)
        plot_metric(acc, f1)
        plot_conf_matrix(tp, tn, fp, fn)
    else:
        plot_loss(train_loss)


def plot_result(shot: int, model: nn.Module, threshold: float,
                data: pd.DataFrame = None):
    """plot prediction and groundtruth

    Args:
        shot (int): number of the shot
        model (nn.model): model
        threshold (float): threshold to predict class
        data (pd.DataFrame, optional): data. Defaults to None.
    """
    if data is None:
        data = pd.read_parquet(os.path.join(DATA_CLEAN,
                                            parquet_name(shot, clean=True)))
    X = data.values.transpose()
    X = torch.from_numpy(X).float()
    X = torch.unsqueeze(X, dim=0)
    model.eval()
    pred = model(X)
    pred = torch.squeeze(pred, dim=1)
    pred = torch.squeeze(pred, dim=0)
    pred = pred.ge(threshold)
    pred = pred.tolist()
    tD = get_tD(shot)
    fig = plt.figure(1, figsize=(14, 3))
    ax1 = plt.subplot()
    if tD == -1000:
        gt = torch.zeros(len(pred)).tolist()
    else:
        tD_start = -1000
        time = data.index.values[-len(pred):]
        f = open(os.path.join(LABEL_PATH, "JET_Train.json"))
        jet_train = json.load(f)
        pres_disr = jet_train['Class']["PreDisr"]['Summary']
        for disr in pres_disr:
            if disr[0] == shot:
                tD_start = disr[1]
        if tD_start == -1000:
            print("no pre distruption time for this shot")
            return
        indTD_Start = np.argmin(np.abs(np.array(time)-tD_start))
        gt = np.zeros(len(pred))
        gt[indTD_Start:] = 1
    ax1.plot(time, gt, color='r', label="groundtruth")
    ax1.plot(time, pred, color='b', label="prediction")
    ax1.set_title("Result prediction")
    ax1.set_ylabel("Label (1 distrpution, 0 non distruption)")
    ax1.set_xlabel("Time")
    fig.tight_layout()
    plt.legend(loc="best")
    plt.show()
