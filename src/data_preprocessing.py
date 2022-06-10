"""data pre-processing
"""
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data_exploration import (check_nan_value, feature_exploaration,
                                  get_tD)
from src.path import DATA_DIR, DATA_PARQUET_PATH, LABEL_PATH, parquet_name


def remove_nan_features(data: pd.DataFrame,
                        threshold: int = 70) -> pd.DataFrame:
    """remove the nan features of the data.

    Args:
        data (pd.DataFrame): data
        threshold (int, optional): threshold for proportion nan to remove the f
        features. Default to 70
    Returns:
        dict: data without nan features
    """
    features_remove = check_nan_value(data, threshold=threshold)
    col_to_drop = []
    for label, _ in features_remove.items():
        col_to_drop.append(label)
    return data.drop(columns=col_to_drop)


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean(0)
    std = data.std(0)
    data = data.subtract(mean).divide(std)
    # constant features
    for features in data.columns[std == 0]:
        data[features].values[:] = 0
    # infinite features
    for features in data.columns[std.isna()]:
        data = data.drop(columns=features)
    return data


def create_label_DB(data: pd.DataFrame,
                    global_DB: pd.DataFrame) -> pd.DataFrame:
    """create label with event of the global table

    Args:
        data (pd.DataFrame): data
        global_DB (pd.DataFrame): distruption event
    """
    data["label"] = 0
    for i, first in zip(range(1, global_DB.shape[1]-2),
                        global_DB.columns.values[1:global_DB.shape[1]-2]):
        if i % 2 == 1:
            index = first.find("first")
            last = first[0:index] + "last"
            label = i//2 + 1
            if ~global_DB[first].isna().values[0]:
                if global_DB[first].values[0] != global_DB[last].values[0]:
                    start = global_DB[first].values[0]
                    end = global_DB[last].values[0]
                    time = data.index.values
                    data["label"].values[np.logical_and(time >= start,
                                                        time <= end)] = label
                else:
                    start = global_DB[first].values[0]
                    time = data.index.values
                    index = np.argmin(np.abs(time-start))
                    data["label"].values[index] = label
    if ~global_DB["tD"].isna().values[0]:
        DB_time = global_DB["tD"].values[0]
        time = data.index.values
        index = np.argmin(np.abs(time-DB_time))
        data["label"].values[index] = 23
    out = data["label"]
    data = data.drop(columns="label")
    return out


def remove_Te_ECE(data: pd.DataFrame) -> pd.DataFrame:
    """remove Te ECE features

    Args:
        data (pd.DataFrame): data to remmove

    Returns:
        pd.DataFrame: data without Te ECE features
    """
    feature_group = feature_exploaration(data)
    nb_ECE_feature = int(feature_group["Te_ECE"]/2)
    col_to_drop = []
    for i in range(1, nb_ECE_feature+1):
        col_to_drop.append("Te_ECE_x" + str(i))
        col_to_drop.append("Te_ECE_z" + str(i))
    return data.drop(columns=col_to_drop)


def remove_Ne_HRTS(data: pd.DataFrame) -> pd.DataFrame:
    """remove Te ECE features

    Args:
        data (pd.DataFrame): data to remmove

    Returns:
        pd.DataFrame: data without Te ECE features
    """
    feature_group = feature_exploaration(data)
    nb_ECE_feature = int(feature_group["Ne_HRTS"]/2)
    col_to_drop = []
    for i in range(1, nb_ECE_feature+1):
        col_to_drop.append("Ne_HRTS_x" + str(i))
        col_to_drop.append("Ne_HRTS_z" + str(i))
    return data.drop(columns=col_to_drop)


def remove_Te_HRTS(data: pd.DataFrame) -> pd.DataFrame:
    """remove Te ECE features

    Args:
        data (pd.DataFrame): data to remmove

    Returns:
        pd.DataFrame: data without Te ECE features
    """
    feature_group = feature_exploaration(data)
    nb_ECE_feature = int(feature_group["Te_HRTS"]/2)
    col_to_drop = []
    for i in range(1, nb_ECE_feature+1):
        col_to_drop.append("Te_HRTS_x" + str(i))
        col_to_drop.append("Te_HRTS_z" + str(i))
    return data.drop(columns=col_to_drop)


def remove_KB5V(data: pd.DataFrame) -> pd.DataFrame:
    """remove Te ECE features

    Args:
        data (pd.DataFrame): data to remmove

    Returns:
        pd.DataFrame: data without Te ECE features
    """
    feature_group = feature_exploaration(data)
    nb_ECE_feature = int(feature_group["KB5V"])
    col_to_drop = []
    for i in range(1, nb_ECE_feature+1):
        col_to_drop.append("KB5V_z" + str(i))
    return data.drop(columns=col_to_drop)


def remove_constant_x(data: pd.DataFrame) -> pd.DataFrame:
    """remove KB5H and KB5V position features because there are constant

    Args:
        data (pd.DataFrame): data to remmove

    Returns:
        pd.DataFrame: data without Te ECE features
    """
    feature_group = feature_exploaration(data)
    nb_KB5H_feature = int(feature_group["KB5H"]/2)
    nb_KB5V_feature = int(feature_group["KB5V"]/2)
    col_to_drop = []
    for i in range(1, nb_KB5H_feature+1):
        col_to_drop.append("KB5H_x" + str(i))
    for i in range(1, nb_KB5V_feature+1):
        col_to_drop.append("KB5V_x" + str(i))
    return data.drop(columns=col_to_drop)


def truncate_features(data: pd.DataFrame,
                      label: pd.DataFrame = 0,
                      end: bool = False) -> pd.DataFrame:
    """truncate the data to remove start and end nan values

    Args:
        data (pd.DataFrame): data
        label (pd.DataFrame): label

    Returns:
        pd.DataFrame: data
        pd.DataFrame: label
    """
    count = 0
    remove = 0
    # start
    for feature in data.columns.values:
        nan = data[feature].isna()
        for t in nan:
            if t:
                count = count + 1
            else:
                if count > remove:
                    remove = count
                count = 0
                break
    data = data.drop(data.index[range(remove)])
    if label:
        label = label.drop(label.index[range(remove)])
    # end
    if end:
        remove = 0
        n = data.shape[0]
        for feature in data.columns.values:
            nan = data[feature].isna()
            for t in reversed(range(n)):
                if nan.values[t]:
                    count = count + 1
                else:
                    if count > remove:
                        remove = count
                    count = 0
                    break
        data = data.drop(data.index[range(n-remove, n)])
    if label:
        label = label.drop(label.index[range(n-remove, n)])
        return data, label
    return data


def interpolate_nan(data: pd.DataFrame) -> pd.DataFrame:
    """interpolate nan value

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: data middle nan values
    """
    for feature in data.columns.values:
        nan_val = data[feature].isna()
        nb_nan = nan_val.sum()
        if nb_nan != 0:
            count = 0
            # manage starting nan
            start = 1
            for i, bool in enumerate(nan_val):
                if bool & start:
                    count = 0
                if ~bool & start:
                    start = 0
                if bool:
                    count += 1
                else:
                    if count != 0:
                        before_val = data[feature].values[i-count-1]
                        after_val = data[feature].values[i]
                        interp_val = (before_val+after_val)/2
                        data[feature].values[range(i-count,  i)] = interp_val
                        count = 0
    return data


def check_nan_inf(data: pd.DataFrame) -> pd.DataFrame:
    """check is there is NAN, inf in data

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: data
    """
    nan = data.isna().sum().any()
    inf = data.isin([np.inf, -np.inf]).sum().any()
    return nan, inf


def IP_feature(data: pd.DataFrame) -> pd.DataFrame:
    """change IP feature =(IPLA - IP)/IP and IPLA

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: data
    """
    data["IP"] = (data["IPLA"] - data["IP"])/data["IP"]
    data = data.drop(columns="IPLA")
    return data


def cut_time_after_distruption(data: pd.DataFrame, shot: int,
                               time: pd.DataFrame,
                               ml: bool = False,
                               drop: float = 150*0.002) -> pd.DataFrame:
    """cut times series after the disruption

    Args:
        data (pd.DataFrame): data
        shot (int): shot_number
        time (pd.DataFrame): time
        ml (bool): use ML time first as final disprution time
        drop (float): drop to cut before the distruption

    Returns:
        pd.DataFrame: data
    """
    tD = get_tD(shot)
    t_cut = tD
    if tD == -1000:
        return data
    elif ml:
        f = open(os.path.join(LABEL_PATH, "JET_Train.json"))
        jet_train = json.load(f)
        for i, s in enumerate(jet_train['ML']["List"]):
            if s == shot:
                t_ML_final = jet_train['ML']["Times_first"][i]
                if t_ML_final < tD:
                    t_cut = t_ML_final
                break
        t_cut_ind = np.argmin(np.abs(np.array(time)-t_cut))
        return data.drop(data.index[range(t_cut_ind, int(data.shape[0]))])
    else:
        t_cut = tD-drop
        t_cut_ind = np.argmin(np.abs(np.array(time)-t_cut))
        return data.drop(data.index[range(t_cut_ind, int(data.shape[0]))])


def clear_nan_distr(data: pd.DataFrame) -> pd.DataFrame:
    """clear channel with nan for distruption shot

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: data
    """
    col_to_drop = data.columns[data.isna().any()]
    return data.drop(columns=col_to_drop)


def clean_shot(shot: int, data: pd.DataFrame = None):
    """clean time series NaN values and missing features for a shot

    Args:
        data (pd.DataFrame): data
        shot (int): number of the shot
    """
    # open parquet
    if data is None:
        data = pd.read_parquet(os.path.join(DATA_PARQUET_PATH,
                                            parquet_name(shot)))
    # remove time
    time = data["time"]
    data.index = time
    data = data.drop(columns="time")
    data = data.drop(columns="ShotNum")

    tD = get_tD(shot)

    # remove Te_ECE data
    data = remove_Te_ECE(data)

    # remove constant feature
    # KB5H and KB5V features
    data = remove_constant_x(data)

    # data = remove_Ne_HRTS(data)
    # data = remove_Te_HRTS(data)
    # data = remove_KB5V(data)

    # cut time after disruption
    if tD != -1000:
        data = cut_time_after_distruption(data, shot, time, ml=False)

    # remove feature with more than 90 % of NAN
    data = remove_nan_features(data, threshold=70)

    # truncate features for beginning NAN and end NAN
    if tD == -1000:
        data = truncate_features(data, end=True)
    else:
        data = truncate_features(data, end=False)
        data = clear_nan_distr(data)

    # manage IP
    # data = IP_feature(data)

    # normalize
    data = normalize_data(data)

    nan, inf = check_nan_inf(data)
    if nan:
        print("Nan in shot: ", shot)

    if inf:
        print("inf in shot: ", shot)
    return data


def label_tD(data: pd.DataFrame, infoDB: pd.DataFrame,
             time_before: float) -> pd.DataFrame:
    """create binary label for distruption for time before the distruption

    Args:
        data (pd.DataFrame): data
        infoDB (pd.DataFrame): info about distruption of the shot
        time_before (float): time before disruption we wish to begin labelling
        as "disruptive"


    Returns:
        pd.DataFrame: label
    """
    label = pd.DataFrame()
    label.index = data.index
    label["DB"] = 0
    if ~infoDB["tD"].isna().values[0]:
        timetD = infoDB["tD"].values
        time = data.index.values
        label["DB"].values[np.logical_and(time > timetD-time_before,
                                          time < timetD)] = 1
    return label


def drop_0D_channel(data: pd.DataFrame):
    """remove 0Dchannel

    Args:
        data (pd.DataFrame): data
    """
    to_remove = ['KB5H_z10', 'KB5H_z11', 'KB5H_z12', 'KB5H_z13', 'KB5H_z14',
                 'KB5H_z15', 'KB5H_z9', 'KB5V_z22', 'KB5V_z5', 'Ne_HRTS_x58',
                 'Ne_HRTS_x59', 'Ne_HRTS_x60', 'Ne_HRTS_x61', 'Ne_HRTS_x62',
                 'Ne_HRTS_x63', 'Ne_HRTS_z1', 'Ne_HRTS_z17', 'Ne_HRTS_z3',
                 'Ne_HRTS_z48', 'Ne_HRTS_z49', 'Ne_HRTS_z50', 'Ne_HRTS_z52',
                 'Ne_HRTS_z53', 'Ne_HRTS_z54', 'Ne_HRTS_z55', 'Ne_HRTS_z56',
                 'Ne_HRTS_z57', 'Ne_HRTS_z58', 'Ne_HRTS_z59', 'Ne_HRTS_z60',
                 'Ne_HRTS_z61', 'Ne_HRTS_z62', 'Ne_HRTS_z63', 'Te_HRTS_x58',
                 'Te_HRTS_x59', 'Te_HRTS_x60', 'Te_HRTS_x61', 'Te_HRTS_x62',
                 'Te_HRTS_x63', 'Te_HRTS_z48', 'Te_HRTS_z49', 'Te_HRTS_z50',
                 'Te_HRTS_z52', 'Te_HRTS_z53', 'Te_HRTS_z54', 'Te_HRTS_z55',
                 'Te_HRTS_z56', 'Te_HRTS_z57', 'Te_HRTS_z58', 'Te_HRTS_z59',
                 'Te_HRTS_z60', 'Te_HRTS_z61', 'Te_HRTS_z62', 'Te_HRTS_z63']
    channel = data.columns.values.tolist()
    col_to_drop = []
    for feature in to_remove:
        if feature in channel:
            col_to_drop.append(feature)
    return data.drop(columns=col_to_drop)


def create_clean_parquet():
    features = ["AREA", "BOLO_HV", "BOLO_V_outer", "BOLO_XDIV", "BTNM", "BVAC",
                "GWfr", "HRTS_Te", "INPWR", "IPLA", "KAPPA", "LI", "MAJRAD",
                "ML", "N1", "N2", "POHM", "PradTot_RT", "Q95", "RGEO", "RMAG",
                "SSXcore", "TBEO", "TOBP", "TOPI", "Zc_v"]
    # get parquet file names
    filenames = os.listdir(DATA_PARQUET_PATH)
    # load each file
    for filename in tqdm(filenames):
        # shot name
        shot_nb = ""
        for m in filename:
            if m.isdigit():
                shot_nb = shot_nb + m
        # data = clean_shot(int(shot_nb))
        data = pd.read_parquet(os.path.join(DATA_PARQUET_PATH, filename))
        x = 1
        data = drop_0D_channel(data)
        feature_group = feature_exploaration(data)
        if feature_group["KB5V"] != 23-2:
            x = 0
        if feature_group["KB5H"] != 24-7:
            x = 0
        if feature_group["Ne_HRTS"] != 126-24:
            x = 0
        if feature_group["Te_HRTS"] != 126-21:
            x = 0
        if feature_group["Te_ECM1"] != 102:
            x = 0
        for feature in features:
            if x == 0:
                break
            data_col = data.columns.values
            for i, other in enumerate(data_col[:feature_group["Other"]]):
                if other == feature:
                    break
                if i == (feature_group["Other"]-1):
                    x = 0
        if x == 1:
            for other in data.columns.values[:feature_group["Other"]]:
                for i, feature in enumerate(features):
                    if other == feature:
                        break
                    if i == len(features)-1:
                        data = data.drop(columns=other)

            # check nan and inf
            nan, inf = check_nan_inf(data)
            if nan:
                dir = os.path.join(DATA_DIR, "clean_parquet_nan")
                data.to_parquet(os.path.join(dir, filename))
            elif inf:
                dir = os.path.join(DATA_DIR, "clean_parquet_inf")
                data.to_parquet(os.path.join(dir, filename))
            else:
                dir = os.path.join(DATA_DIR, "clean_parquet")
                data.to_parquet(os.path.join(dir, filename))
