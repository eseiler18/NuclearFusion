"""Data exploration
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.path import LABEL_PATH

FEATURES_GROUP_NAME = ["KB5V", "KB5H", "Ne_HRTS", "Te_HRTS", "Te_ECM1",
                       "Te_ECE", "Other"]
# x and z features
NUMBER_FEATURES_GROUP = [23 + 23, 24 + 24, 63 + 63, 63 + 63, 96 + 96]


def feature_exploaration(data: pd.DataFrame) -> dict:
    """explorate the features of the data.

    Args:
        data (pd.DataFrame): data to explore

    Returns:
        dict: number of features in different groups.
    """
    features = data.columns.values
    feature_group = dict()
    for group in FEATURES_GROUP_NAME:
        feature_group[group] = 0
    for feature in features:
        for group in FEATURES_GROUP_NAME:
            if (feature.find(group + "_z") != -1 or
                    feature.find(group + "_x") != -1):
                feature_group[group] += 1
                break
            if group == "Other":
                feature_group[group] += 1
                break
    return feature_group


def check_missing_features(features_groups: dict) -> None:
    """Check if there is missing features

    Args:
        fetaures_groups (dict): number of features in each group
    """
    for group, number in zip(FEATURES_GROUP_NAME, NUMBER_FEATURES_GROUP):
        if group == "Other":
            break
        if features_groups[group] != 0 and features_groups[group] != number:
            print("There is {} in group {} but there must be {}".format(
                  features_groups[group], group, number))


def check_nan_value(data: pd.DataFrame, fig: bool = False,
                    threshold: int = 70) -> dict:
    """Check nan value in features and return feature with more
    than threshold % of nan value

    Args:
        data (pd.DataFrame): data
        fig (bool, optional): boolean for figure. Defaults to False.
        threshold (int, optional): percentage of missing value to return
        the feature. Defaults to 70.

    Returns:
        dict: feature with more than threshold missing values
    """
    propNAN = data.isna().sum().divide(data.shape[0]/100)
    labelNAN = propNAN.index[propNAN.values > threshold].values
    out = dict()
    for label in labelNAN:
        out[label] = propNAN[label]
    if fig:
        fig = plt.figure(1)
        ax1 = plt.subplot(211)
        plt.boxplot(propNAN)
        ax1.set_title("Boxplot of NAN value")
        ax1.set_ylabel("proportion %")

        ax2 = plt.subplot(212)
        plt.plot(propNAN)
        ax2.set_title("NAN value")
        ax2.set_xlabel("Label")
        ax2.set_ylabel("proportion %")
        plt.xticks([])

        fig.tight_layout()
        plt.show()
    return out


def get_distruption_info(shot: int) -> pd.DataFrame:
    """Get the distruption information for a selected shot

    Args:
        shot (int): number of the shot XXXXX

    Returns:
        pd.DataFrame: information concerning distruption
        RU, RD, FT, TD, CQ_80_20
    """
    label = pd.read_excel(os.path.join(LABEL_PATH, "MasterDBtable.xlsx"))
    out = label[label.iloc[:, 0] == shot]
    return out


def get_distruption_global_info(shot: int) -> pd.DataFrame:
    """Get the gloabl ofdistruption information for a selected shot

    Args:
        shot (int): number of the shot XXXXX

    Returns:
        pd.DataFrame: gloabl information concerning distruption
    """
    label = pd.read_csv(os.path.join(LABEL_PATH, "Global_Summary_Table.csv"),
                        delimiter=';')
    out = label[label.iloc[:, 0] == shot]
    return out


def get_tD(shot: int) -> float:
    """get disruption time -1000 is no disruption

    Args:
        shot (int): shot number

    Returns:
        float: tD
    """
    info = get_distruption_global_info(shot)
    if info["tD"].isna().any():
        return -1000
    else:
        return info["tD"].values[0]
