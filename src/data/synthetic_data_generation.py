from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from non_parametric import frequency_table
from non_parametric.empirical_cumulative_distribution import cdf
from sklearn.base import BaseEstimator, TransformerMixin

def generate_multivariate_data(
    X: pd.DataFrame, bins: int = 10, N: int = 1000
) -> pd.DataFrame:
    """
    Generate synthetic multivariate data preserving dependency structures between variables.

    This implementation is based on the method proposed in:

    Restrepo, J.P.; Rivera, J.C.; Laniado, H.; Osorio, P.; Becerra, O.A.
    "Nonparametric Generation of Synthetic Data Using Copulas."
    Electronics 2023, 12, 1601.
    https://doi.org/10.3390/electronics12071601

    The original implementation was adapted from the `non_parametric` Python package.
    Modifications were introduced to ensure compatibility with recent versions of
    pandas (>= 2.0), specifically addressing deprecated indexing behavior
    (e.g., replacing label-based indexing with positional indexing using `.iloc`).

    Parameters
    ----------
    X : pd.DataFrame
        Original dataset used as reference for generating synthetic data.
    bins : int, default=10
        Number of intervals (bins) used to construct frequency tables
        for each variable.
    N : int, default=1000
        Number of synthetic samples to generate.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with the same columns as `X`, preserving
        the dependency structure between variables.

    Notes
    -----
    The method works by:
    1. Estimating empirical cumulative distribution functions (CDFs)
    for each variable.
    2. Building frequency tables based on discretized intervals.
    3. Sampling from the empirical joint distribution using copula-based logic.
    4. Generating new samples via uniform sampling within selected intervals.

    Important
    ---------
    This version includes fixes for compatibility with modern pandas versions.
    The original implementation may raise errors (e.g., KeyError) due to changes
    in indexing behavior introduced in pandas >= 2.0.
    """

    # i) generate matrix of empirical distributions

    matrix_F = X.copy(deep=True)

    for i in matrix_F.columns:
        X_column_i = matrix_F[i]
        x_sort_i, F_i = cdf(X_column_i)
        matrix_F[i] = [F_i[np.where(x_sort_i == z)[0][0]] for z in X_column_i]

    # ii) A frequency table is constructed for each variable with
    # the given number of bins.

    dicc_freq_tables = {}

    for i in X.columns:
        X_column_i = X[i]
        simple_table = frequency_table(X_column_i, bins=bins)
        complete_table = pd.DataFrame.from_dict(
            simple_table, orient="index", columns=["Freq_abs"]
        )
        freq_rel = [j / len(X_column_i) for j in simple_table.values()]
        complete_table["Freq_rel"] = freq_rel
        complete_table["Freq_acum"] = np.cumsum(freq_rel)

        dicc_freq_tables[i] = complete_table

    # iii) - iv)  List of N integers between 0 and n-1

    list_N = np.random.randint(low=0, high=len(matrix_F), size=N)

    # v) - vi)  Simulation

    X_generated = pd.DataFrame(columns=X.columns)
    for sub_n in list_N:
        random_generated = []

        for i in X.columns:
            h = matrix_F.loc[sub_n, i]
            # inverval or freq_table where is the percentile
            interval = next(
                (
                    j
                    for j in range(0, len(dicc_freq_tables[i]["Freq_acum"]))
                    if dicc_freq_tables[i]["Freq_acum"].iloc[j] >= (h)
                ),
                None,
            )
            if interval == None:
                interval = -1

            lim_inf = dicc_freq_tables[i].index[interval][0]
            lim_sup = dicc_freq_tables[i].index[interval][1]
            random_generated.append(
                np.random.uniform(low=lim_inf, high=lim_sup, size=1)[0]
            )

        random_generated = np.array(random_generated).T
        random_generated = pd.DataFrame([random_generated], columns=X.columns)
        X_generated = pd.concat([X_generated, random_generated], ignore_index=True)

    return X_generated



class SyntheticDataGenerator(BaseEstimator, TransformerMixin):
    """
    Synthetic data generator compatible with sklearn pipelines.

    Generates synthetic samples during transform using stored y from fit.
    """

    def __init__(self, n_to_generate: Optional[Dict] = None):
        self.n_to_generate = n_to_generate

    def _calculate_samples_to_generate(self, y: pd.Series) -> Dict:
        values_dict = y.value_counts().to_dict()
        max_count = max(values_dict.values())

        return {k: max_count - v for k, v in values_dict.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.y_ = y.reset_index(drop=True)

        if self.n_to_generate is None:
            self.n_to_generate_ = self._calculate_samples_to_generate(self.y_)
        else:
            self.n_to_generate_ = self.n_to_generate

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        y = self.y_

        synthetic_X_list = []

        for key, n in self.n_to_generate_.items():
            if n > 0:
                X_class = X[y == key].copy().reset_index(drop=True)

                synthetic_X = generate_multivariate_data(X=X_class, N=n)
                synthetic_X_list.append(synthetic_X)

        if synthetic_X_list:
            X_synth = pd.concat(synthetic_X_list, ignore_index=True)
            X_res = pd.concat([X, X_synth], ignore_index=True)
        else:
            X_res = X

        return X_res

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        self.fit(X, y)

        X_res = self.transform(X)

        y_res = self._resample_y(y)

        return X_res, y_res

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:

        self.fit(X, y)

        X_res = self.transform(X)
        y_res = self._resample_y(y)

        return X_res, y_res

    def _resample_y(self, y: pd.Series) -> pd.Series:
        synthetic_y_list = []

        for key, n in self.n_to_generate_.items():
            if n > 0:
                synthetic_y_list.append(pd.Series([key] * n))

        if synthetic_y_list:
            y_synth = pd.concat(synthetic_y_list, ignore_index=True)
            y_res = pd.concat([y.reset_index(drop=True), y_synth], ignore_index=True)
        else:
            y_res = y

        return y_res
