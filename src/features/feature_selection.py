from typing import List, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection transformer compatible with scikit-learn pipelines.

    This transformer performs a two-step feature selection process:

    1. Correlation-based filtering:
       Removes features that are highly correlated with others based on a
       specified threshold to reduce multicollinearity.

    2. Mutual information ranking:
       Selects the top-k most informative features with respect to the target
       variable using mutual information.

    This approach ensures robustness across datasets by:
    - Reducing redundancy when high correlation exists
    - Falling back to relevance-based selection when it does not

    Parameters
    ----------
    corr_threshold : float, default=0.9
        Threshold for absolute Pearson correlation. Features with correlation
        above this value are considered redundant and removed.

    top_k : int or None, default=10
        Number of top features to keep based on mutual information.
        If None, all features are retained after correlation filtering.

    random_state : int, default=42
        Random seed used in mutual information estimation for reproducibility.

    Attributes
    ----------
    selected_features_ : List[str]
        List of selected feature names after applying both filtering steps.

    removed_corr_features_ : List[str]
        List of features removed due to high correlation.
    """

    def __init__(
        self,
        corr_threshold: float = 0.9,
        top_k: Optional[int] = 10,
        random_state: int = 42,
    ):
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.removed_corr_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector to the training data.

        This method computes:
        - Features to remove based on correlation
        - Feature importance scores using mutual information

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        y : pd.Series
            Target variable.

        Returns
        -------
        self : FeatureSelector
            Fitted instance of the transformer.
        """
        X = X.copy()

        # Correlation filtering
        corr_matrix = X.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.corr_threshold)
        ]

        self.removed_corr_features_ = to_drop

        X_filtered = X.drop(columns=to_drop, errors="ignore")

        # Mutual Information
        mi = mutual_info_classif(X_filtered, y, random_state=self.random_state)

        mi_series = pd.Series(mi, index=X_filtered.columns).sort_values(ascending=False)

        if self.top_k is not None:
            self.selected_features_ = mi_series.head(self.top_k).index.tolist()
        else:
            self.selected_features_ = mi_series.index.tolist()

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform the dataset by selecting the previously chosen features.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Transformed dataset containing only selected features.
        """
        return X[self.selected_features_]

    def get_feature_names_out(self):
        """
        Get the names of the selected output features.

        Returns
        -------
        List[str]
            List of selected feature names.
        """
        return self.selected_features_
