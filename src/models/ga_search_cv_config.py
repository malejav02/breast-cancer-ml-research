from typing import Union
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from models.model_settings import ModelConfig, FeatureSelectorConfig
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics._scorer import _BaseScorer
import mlflow.data
import copy
# from data.synthetic_data_generation import SyntheticDataGenerator
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict
from evaluation.metrics import ClassificationMetrics
from models.export_model import export_model
import mlflow
import mlflow.sklearn
import time
import numpy as np
import sklearn
from utils.seed import set_seed
import datetime

callbacks = [ConsecutiveStopping(generations=5, metric="fitness")]


class GAModelSearch:
    """
    Genetic Algorithm-based hyperparameter optimization across multiple classifiers,
    including preprocessing pipelines, evaluation, and full experiment tracking using MLflow.

    Attributes
    ----------
    estimators : dict
        Dictionary of model names and corresponding estimator instances.
    class_weight_classifiers : list
        Models that support class_weight and do not require synthetic balancing.
    classifiers_without_class_weight : list
        Models that require synthetic data balancing.
    """

    def __init__(self) -> None:
        """Initialize estimators and categorize models based on balancing requirements."""

        self.estimators = ModelConfig().estimators
        self.class_weight_classifiers = [
            "DecisionTreeClassifier",
            "ExtraTreesClassifier",
            "LogisticRegression",
            "RandomForestClassifier",
            "SVC",
            "LGBMClassifier",
        ]
        self.classifiers_without_class_weight = [
            "GaussianNB",
            "KNeighborsClassifier",
            "LinearDiscriminantAnalysis",
            "QuadraticDiscriminantAnalysis",
            "XGBClassifier",
        ]

    @staticmethod
    def add_prefix_to_params(param_dict, prefix):
        return {f"{prefix}__{k}": v for k, v in param_dict.items()}

    def genopt_training(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: Union[BaseCrossValidator, int],
        scoring: Union[_BaseScorer, str],
        population_size: int = 20,
        generations: int = 50,
        save_model: bool = False,
        data_name: str = "wisconsin",
    ) -> None:
        """
        Perform genetic hyperparameter optimization across multiple models.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target variable.
        X_test : pd.DataFrame
            Testing feature matrix.
        y_test : pd.Series
            Testing target variable.
        cv : int or BaseCrossValidator
            Cross-validation strategy.
        scoring : str or scorer
            Optimization metric.
        population_size : int, optional
            Population size for the genetic algorithm.
        generations : int, optional
            Number of generations for the genetic algorithm.
        save_model : bool, optional
            Whether to export the best model to disk.
        data_name : str, optional
            Dataset identifier for tracking and export purposes.

        Returns
        -------
        None
        """
        best_score = float("-inf")
        best_model = None
        best_model_name = ""
        best_run_id = None
        best_metrics_dict = {}

        selector_config = FeatureSelectorConfig()
        selector_dict = selector_config.get_selector_config()

        selector = selector_dict["selector"]
        selector_space = GAModelSearch.add_prefix_to_params(
            selector_dict["search_space"], prefix="feature_selection"
        )

        # Loop through each model type
        for name, estimator in self.estimators.items():

            with mlflow.start_run(
                run_name=f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
                nested=True,
            ):
                current_run_id = mlflow.active_run().info.run_id

                print(f"Performing evolutionary optimization for {name}...")

                model_confing = ModelConfig(estimator_name=name)
                param_grid = GAModelSearch.add_prefix_to_params(
                    param_dict=model_confing.get_hyperparameter_search_space(),
                    prefix="estimator",
                )
                param_grid.update(selector_space)

                mlflow.set_tag("dataset_name", data_name)

                df_train = X_train.copy()
                df_train["target"] = y_train

                mlflow.log_input(
                    mlflow.data.from_pandas(
                        df=df_train, targets="target", name=data_name
                    ),
                    context="training",
                )

                mlflow.log_param("n_samples", X_train.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])

                if name in self.class_weight_classifiers:
                    pipeline = Pipeline(
                        [
                            ("feature_selection", selector),
                            ("scaler", StandardScaler()),
                            ("estimator", estimator),
                        ]
                    )
                elif name in self.classifiers_without_class_weight:
                    pipeline = Pipeline(
                        [
                            ("feature_selection", selector),
                            # ("balancer", SyntheticDataGenerator()),
                            ("balancer", SMOTE(random_state=set_seed())),
                            ("scaler", StandardScaler()),
                            ("estimator", estimator),
                        ]
                    )
                pipeline.set_output(transform="pandas")

                # Configure the search
                mlflow.log_param("cv", str(cv))
                mlflow.log_param("scoring", str(scoring))
                mlflow.log_param("population_size", population_size)
                mlflow.log_param("generations", generations)
                start = time.time()
                evolved_estimator = GASearchCV(
                    estimator=pipeline,
                    cv=cv,
                    scoring=scoring,
                    param_grid=param_grid,
                    n_jobs=-1,
                    verbose=True,
                    population_size=population_size,
                    generations=generations,
                ).fit(X_train, y_train, callbacks=callbacks)
                end = time.time()

                score = evolved_estimator.best_score_
                best_pipeline = evolved_estimator.best_estimator_
                inference_pipeline = Pipeline(
                    [
                        ("scaler", best_pipeline.named_steps["scaler"]),
                        ("estimator", best_pipeline.named_steps["estimator"]),
                    ]
                )
                selected_features = best_pipeline.named_steps["scaler"].feature_names_in_

                print(f"Best {name} model achieved score: {score:.2f}")

                mlflow.log_params(evolved_estimator.best_params_)
                mlflow.log_metric("cv_score", score)

                y_pred = cross_val_predict(
                    inference_pipeline,
                    X_train[selected_features],
                    y_train,
                    cv=cv,
                    method="predict",
                )
                y_proba = cross_val_predict(
                    inference_pipeline,
                    X_train[selected_features],
                    y_train,
                    cv=cv,
                    method="predict_proba",
                )[:, 1]
                metrics = ClassificationMetrics(
                    y_true=y_train, y_pred=y_pred, y_proba=y_proba
                )
                metrics_dict = metrics.get_metrics()

                mlflow.log_metrics(metrics_dict)
                mlflow.set_tag("model_name", name)
                mlflow.set_tag("model_type", type(estimator).__name__)
                mlflow.set_tag("selected_features", selected_features)
                mlflow.sklearn.log_model(
                    inference_pipeline.named_steps["scaler"],
                    name="scaler",
                    serialization_format="pickle",
                    pip_requirements="requirements.txt",
                )
                mlflow.sklearn.log_model(
                    inference_pipeline.named_steps["estimator"],
                    name="estimator",
                    serialization_format="pickle",
                    pip_requirements="requirements.txt",
                )

                mlflow.set_tag(
                    "uses_balancer", name in self.classifiers_without_class_weight
                )
                mlflow.set_tag("uses_scaling", True)
                mlflow.set_tag("evaluation_method", "cross_val_predict")
                mlflow.set_tag("cv_folds", str(cv))
                mlflow.set_tag("prediction_type", "out_of_fold")
                conf_matrix = metrics.plot_confusion_matrix()
                class_report = metrics.plot_classification_report()

                mlflow.log_figure(conf_matrix, "confusion_matrix_cross_val_predict.png")
                mlflow.log_figure(
                    class_report, "classification_report_cross_val_predict.png"
                )
                mlflow.log_metric("training_time_sec", end - start)
                mlflow.set_tag("sklearn_version", sklearn.__version__)
                mlflow.set_tag("numpy_version", np.__version__)

                # Update best model if current model is better
                if score > best_score:
                    best_score = score
                    best_model = copy.deepcopy(inference_pipeline)
                    best_model_name = name
                    best_metrics_dict = metrics_dict
                    best_run_id = current_run_id

        if best_model is not None:
            with mlflow.start_run(run_id=best_run_id):
                print("Logging test metrics into best model run...")

                X_test_selected = X_test[best_model.feature_names_in_]

                y_pred_test = best_model.predict(X_test_selected)
                y_proba_test = best_model.predict_proba(X_test_selected)[:, 1]

                test_metrics = ClassificationMetrics(
                    y_true=y_test, y_pred=y_pred_test, y_proba=y_proba_test
                )

                test_metrics_dict = test_metrics.get_metrics()

                mlflow.log_metrics(
                    {f"test_{k}": v for k, v in test_metrics_dict.items()}
                )

                mlflow.set_tag("test_evaluation", True)
                df_test = X_test.copy()
                df_test["target"] = y_test

                mlflow.log_input(
                    mlflow.data.from_pandas(
                        df=df_test, targets="target", name=data_name
                    ),
                    context="test",
                )

            if save_model:
                model_dict = {
                    "model": best_model,
                    "name": best_model_name,
                    "selected_features": best_model.feature_names_in_,
                    "hyperparameters": best_model.get_params(),
                    "best_score_cv(f1_macro)": best_score,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "f1_macro_test": test_metrics_dict["f1_macro"],
                }
                model_dict.update(best_metrics_dict)
                print(
                    f"Exporting best model ({best_model_name}) with score: {best_score:.2f}"
                )
                model_dict.update(metrics_dict)
                export_model(
                    model_dict=model_dict, model_name=f"{data_name}_best_model.pkl"
                )
