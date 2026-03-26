from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
import pandas as pd
from utils.seed import set_seed


class DataSplitter:
    """
    Data splitter for train/test or train/val/test.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = set_seed(random_state)

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify: bool = True,
    ) -> Tuple:
        """
        Split data into train/test or train/val/test.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        test_size : float
            Proportion of data for test set
        val_size : float, optional
            Proportion of training data to use as validation set. If None, only train/test split
        stratify : bool
            Whether to stratify by target

        Returns
        -------
        If val_size is None: X_train, X_test, y_train, y_test
        Else: X_train, X_val, X_test, y_train, y_val, y_test
        """
        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param,
        )

        if val_size is not None:
            stratify_train = y_train if stratify else None
            val_relative = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_relative,
                random_state=self.random_state,
                stratify=stratify_train,
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

        return X_train, X_test, y_train, y_test
