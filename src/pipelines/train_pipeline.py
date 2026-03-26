import mlflow
from sklearn.model_selection import StratifiedKFold
from data.load_data import load_wisconsin_dataset
from data.split_data import DataSplitter
from models.ga_search_cv_config import GAModelSearch
from utils.seed import set_seed

def run_training_pipeline(n_splits:int = 5):

    # Load data
    X, y = load_wisconsin_dataset(save_local=True)

    # Split data
    splitter = DataSplitter(random_state=set_seed())

    X_train, X_test, y_train, y_test = splitter.split(
        X, y,
        test_size=0.2,
        stratify=True
    )

    # MLflow setup
    mlflow.set_experiment("Wisconsin_Breast_Cancer_Classification")

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = set_seed())

    #  Genetic optimization
    search = GAModelSearch()

    search.genopt_training(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=cv,
        scoring="f1_macro",
        population_size=15,
        generations=100,
        save_model=True,
        data_name="wisconsin",
    )


if __name__ == "__main__":
    run_training_pipeline()
