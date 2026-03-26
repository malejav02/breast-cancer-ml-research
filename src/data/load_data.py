"""
Data loading utilities for breast cancer datasets.

This module provides functions to download and load the following datasets:

1. Breast Cancer Coimbra Dataset (UCI)
2. Breast Cancer Wisconsin Diagnostic Dataset (sklearn)
3. BreakHis histopathological image dataset (directory preparation)

Datasets are saved in the /data/raw directory outside the src folder.

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any
from io import StringIO

import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer
from utils.paths import data_raw_dir


COIMBRA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv"
)


def load_coimbra_dataset(download: bool = True) -> pd.DataFrame:
    """
    Load the Breast Cancer Coimbra dataset.

    Parameters
    ----------
    download : bool
        If True, downloads the dataset if it does not exist locally.

    Returns
    -------
    pd.DataFrame
        Coimbra dataset as a pandas DataFrame.
    """

    file_path = data_raw_dir("coimbra.csv")

    if download and not file_path.exists():

        response = requests.get(COIMBRA_URL, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        df.to_csv(file_path, index=False)

    else:
        df = pd.read_csv(file_path)

    return df


def load_wisconsin_dataset(save_local: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Breast Cancer Wisconsin Diagnostic dataset.

    Parameters
    ----------
    save_local : bool
        If True, saves the dataset as a CSV file in the data directory.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """

    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    if save_local:
        df = pd.concat([X, y], axis=1)
        df.to_csv(data_raw_dir("wisconsin.csv"), index=False)

    return X, y


def prepare_breakhis_directory() -> Path:
    """
    Prepare the directory for the BreakHis dataset.

    BreakHis cannot be automatically downloaded due to licensing and size.
    Users must manually place the dataset inside the folder created here.

    Returns
    -------
    Path
        Path to the BreakHis dataset directory.
    """

    breakhis_dir = data_raw_dir("breakhis")
    breakhis_dir.mkdir(parents=True, exist_ok=True)

    return breakhis_dir


def load_all_datasets() -> Dict[str, Any]:
    """
    Download and load all datasets used in the project.

    This function ensures that:
    - Coimbra dataset is downloaded
    - Wisconsin dataset is saved locally
    - BreakHis directory exists

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all loaded datasets.
    """

    print("Loading Coimbra dataset...")
    coimbra = load_coimbra_dataset()

    print("Loading Wisconsin dataset...")
    wisconsin_X, wisconsin_y = load_wisconsin_dataset()

    print("Preparing BreakHis directory...")
    breakhis_path = prepare_breakhis_directory()

    print("All datasets prepared successfully.")

    return {
        "coimbra": coimbra,
        "wisconsin_X": wisconsin_X,
        "wisconsin_y": wisconsin_y,
        "breakhis_path": breakhis_path,
    }


if __name__ == "__main__":
    datasets = load_all_datasets()
