from typing import Dict
from utils.paths import models_dir
import joblib


def export_model(
    model_dict: Dict,
    model_name: str,
) -> None:
    """
    Serialize and save a trained model along with its metadata to disk.

    This function stores a dictionary containing the trained model and
    associated information (e.g., hyperparameters, training data, metrics)
    using joblib.

    Parameters
    ----------
    model_dict : Dict
        Dictionary containing the model and related metadata.

    model_name : str
        Name of the output file. This is also used to determine the storage path via `models_dir`.

    Returns
    -------
    None
        The function does not return anything. It saves the model to disk.

    """

    models_DIR = models_dir(model_name)
    joblib.dump(model_dict, models_DIR)
