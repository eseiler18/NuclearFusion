"""
Paths and archives management.
"""
import os
import time


# Directories paths
DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DIRNAME)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUT_DIR = os.path.join(ROOT_DIR, 'output')

# Dataset paths
DATA_PARQUET_PATH = os.path.join(DATA_DIR, 'parquet')
DATA_CLEAN = os.path.join(DATA_DIR, 'clean_parquet')
LABEL_PATH = os.path.join(DATA_DIR, 'label')

# Training paths
DEFAULT_LOSSES_PATH = os.path.join(MODELS_DIR, 'losses.pickle')
DEFAULT_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'weights.pt')
DEFAULT_PARAMETERS_PATH = os.path.join(MODELS_DIR, 'parameters.pickle')

# Testing paths
DEFAULT_PREDICTIONS_DIR = os.path.join(OUT_DIR, 'predictions')


def create_dirs() -> None:
    """Creates directories if needed."""
    for path in (MODELS_DIR, OUT_DIR):
        if not os.path.exists(path):
            os.mkdir(path)


def generate_filename(filename: str) -> str:
    """Generates a filename using time.

    Args:
        filename (str): string template. Example: 'model-{}.pt'.

    Returns:
        str: filename with time.
    """
    timestr = time.strftime('%d%m%Y')
    return filename.format(timestr)


def generate_model_filename(model: str, epoch: int, channel: int) -> str:
    """Generates model filename using time.

    Args:
        model (str): name model.
        epoch (int): number of epochs trained
    Returns:
        str: model filename with time.
    """
    filename = generate_filename('-{}.pt')
    filename = model + f'-{channel}chan-{epoch}epoch' + filename
    return os.path.join(OUT_DIR, filename)


def generate_log_filename(epoch: int, channel: int) -> str:
    """Generates log filename using time.

    Args:
        epoch (int): number of epochs trained
    Returns:
        str: log filename with time.
    """
    filename = generate_filename('-{}.pickle')
    filename = f'log-{channel}chan-{epoch}epoch' + filename
    return os.path.join(OUT_DIR, filename)


def generate_result_filename(model: str) -> str:
    """Generates log filename using time.

    Args:
        model (str): model name
    Returns:
        str: result filename.
    """
    filename = "result-" + model[:-2] + "pickle"
    return os.path.join(OUT_DIR, filename)


def parquet_name(number: int, clean: bool = False) -> str:
    """return the parquet data filname with the corresponding number

    Args:
        number (int): number of the parquet data in the dataset
        clean (bool): if have to had clean to the name

    Returns:
        str: name of the parquet data
    """
    if clean:
        return "clean" + "JETno" + str(number) + ".parquet"
    return "JETno" + str(number) + ".parquet"
