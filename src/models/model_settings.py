from typing import Dict, Any
from sklearn_genetic.space import Integer, Continuous, Categorical
from features.feature_selection import FeatureSelector
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import lightgbm as lgb
from utils.seed import set_seed


class ModelConfig:

    def __init__(self, estimator_name: str = "SVC") -> None:
        """
        Initialize the model configuration object.

        This class provides a unified interface to access machine learning
        estimators and their corresponding hyperparameter search spaces used
        for optimization procedures.

        Parameters
        ----------
        estimator_name : str, default="SVC"
            Name of the estimator whose configuration will be used. The name must
            correspond to one of the supported estimators defined in the internal
            registry.

            Supported estimators are:

            - "DecisionTreeClassifier"
            - "ExtraTreesClassifier"
            - "GaussianNB"
            - "KNeighborsClassifier"
            - "LinearDiscriminantAnalysis"
            - "LogisticRegression"
            - "QuadraticDiscriminantAnalysis"
            - "RandomForestClassifier"
            - "SVC"
            - "XGBClassifier"
            - "LGBMClassifier"

        Raises
        ------
        ValueError
            If the provided estimator name is not supported.

        """

        self.estimators = {
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "ExtraTreesClassifier": ExtraTreesClassifier(),
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
            "LogisticRegression": LogisticRegression(),
            "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
            "RandomForestClassifier": RandomForestClassifier(),
            "SVC": SVC(),
            "XGBClassifier": xgb.XGBClassifier(),
            "LGBMClassifier": lgb.LGBMClassifier(),
        }
        self.estimator_name = estimator_name
        if self.estimator_name not in self.estimators.keys():
            raise ValueError(
                f"Unsupported estimator name: {estimator_name}. Choose from: {list(self.estimators.keys())}"
            )

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Return the hyperparameter search space for a given classifier.

        This function defines hyperparameter search spaces for multiple classifiers.
        These spaces are intended to be used with genetic algorithm optimization methods.


        Returns
        -------
        Dict[str, Any]
            A dictionary where the keys correspond to hyperparameter names and the
            values define the search space.

            This dictionary can be directly used in genetic hyperparameter
            optimization procedures.

        """

        hyperparameter_spaces = {
            "DecisionTreeClassifier": {
                "criterion": Categorical(["gini", "entropy", "log_loss"]),
                "max_depth": Integer(5, 20),
                "min_samples_split": Continuous(0.05, 1),
                "min_samples_leaf": Continuous(0.05, 1),
                "max_features": Categorical(choices=["sqrt", "log2"]),
                "class_weight": Categorical(choices=["balanced"]),
            },
            "ExtraTreesClassifier": {
                "n_estimators": Integer(50, 500),
                "criterion": Categorical(choices=["gini", "entropy", "log_loss"]),
                "max_depth": Integer(5, 50),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 5),
                "max_features": Categorical(choices=["sqrt", "log2", None]),
                "bootstrap": Categorical(choices=[True, False]),
                "class_weight": Categorical(["balanced"]),
                "ccp_alpha": Continuous(0, 1),
            },
            "GaussianNB": {"var_smoothing": Continuous(1e-9, 1e-3)},
            "KNeighborsClassifier": {
                "n_neighbors": Integer(3, 15),
                "weights": Categorical(["uniform", "distance"]),
                "metric": Categorical(
                    ["euclidean", "manhattan", "minkowski", "cosine"]
                ),
                "leaf_size": Integer(5, 100),
                "p": Integer(1, 2),
            },
            "LinearDiscriminantAnalysis": {
                "solver": Categorical(choices=["svd", "lsqr", "eigen"]),
                "tol": Continuous(0, 1),
            },
            "LogisticRegression": {
                "tol": Continuous(1e-9, 1e-1),
                "fit_intercept": Categorical(choices=[True, False]),
                "intercept_scaling": Continuous(0, 1),
                "class_weight": Categorical(choices=["balanced"]),
                "C": Continuous(0.00001, 1),
                "solver": Categorical(
                    choices=[
                        "lbfgs",
                        "liblinear",
                        "saga",
                        "newton-cg",
                        "newton-cholesky",
                        "sag",
                    ]
                ),
            },
            "QuadraticDiscriminantAnalysis": {
                "reg_param": Continuous(0, 0.5),
                "tol": Continuous(1e-5, 1e-3),
            },
            "RandomForestClassifier": {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(10, 50),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 5),
                "bootstrap": Categorical([True, False]),
                "class_weight": Categorical(["balanced"]),
            },
            "SVC": {
                "C": Continuous(0.0001, 100),
                "gamma": Categorical(["scale", "auto"]),
                "kernel": Categorical(["rbf", "linear"]),
                "tol": Continuous(1e-5, 1e-3),
                "class_weight": Categorical(choices=["balanced"]),
                "max_iter": Integer(50, 500),
                "probability": Categorical([True]),
            },
            "XGBClassifier": {
                "n_estimators": Integer(50, 500),
                "learning_rate": Continuous(0.0001, 0.5),
                "max_depth": Integer(2, 10),
                "subsample": Continuous(0, 1),
                "gamma": Continuous(0, 1),
            },
            "LGBMClassifier": {
                "boosting_type": Categorical(choices=["gbdt"]),
                "n_estimators": Integer(50, 500),
                "learning_rate": Continuous(0.0001, 0.5),
                "num_leaves": Integer(10, 50),
                "class_weight": Categorical(choices=["balanced"]),
            },
        }

        grid = hyperparameter_spaces[self.estimator_name]

        return grid

    def get_estimator_config(self) -> Dict[str, Any]:
        """
        Retrieve the configuration associated with the selected estimator.

        This method returns a dictionary containing the instantiated estimator
        object and its corresponding hyperparameter search space.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the estimator configuration with the following keys:

            - "estimator" : BaseEstimator
                Instantiated machine learning estimator compatible with the
                scikit-learn API.

            - "search_space" : Dict[str, Any]
                Hyperparameter search space defining the parameters to optimize.
                The values correspond to search space objects such as
                ``Integer``, ``Continuous``, or ``Categorical``.
        """

        estimator_config_dict = {
            "estimator": self.estimators[self.estimator_name](),
            "search_space": self.get_hyperparameter_search_space(),
        }

        return estimator_config_dict


class FeatureSelectorConfig:
    """
    Configuration class for feature selection components.

    This class provides a unified interface to retrieve both the feature selector
    instance and its corresponding hyperparameter search space, enabling seamless
    integration with optimization pipelines.

    Parameters
    ----------
    selector_name : str, default="FeatureSelector"
        Name of the feature selector to be used.

        Currently supported:
        - "FeatureSelector"

    Raises
    ------
    ValueError
        If an unsupported selector name is provided.
    """

    def __init__(self, selector_name: str = "FeatureSelector") -> None:

        self.selectors = {"FeatureSelector": FeatureSelector(random_state=set_seed())}

        self.selector_name = selector_name

        if self.selector_name not in self.selectors:
            raise ValueError(
                f"Unsupported selector name: {selector_name}. "
                f"Choose from: {list(self.selectors.keys())}"
            )

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Return the hyperparameter search space for the selected feature selector.

        Returns
        -------
        Dict[str, Any]
            Dictionary defining the search space for each hyperparameter.
            Compatible with genetic optimization frameworks.
        """

        hyperparameter_spaces = {
            "FeatureSelector": {
                "corr_threshold": Continuous(0.85, 1),
                "top_k": Integer(15, 20),
            }
        }

        return hyperparameter_spaces[self.selector_name]

    def get_selector_config(self) -> Dict[str, Any]:
        """
        Retrieve the configuration associated with the selected feature selector.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:

            - "selector": Transformer instance
            - "search_space": Hyperparameter search space
        """

        selector_class = self.selectors[self.selector_name]

        selector_config_dict = {
            "selector": selector_class,
            "search_space": self.get_hyperparameter_search_space(),
        }

        return selector_config_dict
