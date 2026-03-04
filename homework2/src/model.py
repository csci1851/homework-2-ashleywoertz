"""
Model stencil for Homework 2: Ensemble Methods with Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Union
from sklearn.tree import plot_tree
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate,GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set plotting style
sns.set_style("whitegrid")


class GradientBoostingModel:
    """Gradient Boosting model implementation with comprehensive evaluation and analysis tools"""

    def __init__(
        self,
        task: str = "classification",
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 50,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
        use_scaler: bool = False,
    ):
        """
        Initialize Gradient Boosting model with customizable parameters

        Args:
            task: 'classification' or 'regression'
            max_depth: Maximum depth of a tree (controls pruning)
            learning_rate: Step size shrinkage to prevent overfitting
            n_estimators: Number of boosting rounds/trees
            subsample: Subsample ratio of training instances
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random seed for reproducibility
            use_scaler: Whether to apply StandardScaler before training/prediction
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")

        self.model = None
        self.feature_names = None
        self.task = task
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: Optional[pd.Series] = None
    ):
        """
        Split data into training and testing sets

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        # handle unbalanced data (for the cancer set specifically) by adding stratify!
        if stratify is not None:
            return train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
        else:
            return train_test_split(X, y, test_size = test_size, random_state = random_state)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True):
        """
        Train the Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training targets
            verbose: Whether to print training progress

        Returns:
            self: Trained model instance
        """
        # CREATE MODEL
        if self.task == "classification":
            self.model = GradientBoostingClassifier(**self.params)
        else:
            self.model = GradientBoostingRegressor(**self.params)

        self.feature_names = X_train.columns

        # FIT MODEL
        if self.use_scaler == True:
            X_train = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train, y_train)

        return self

    def predict(
        self, X: pd.DataFrame, return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the trained model

        Args:
            X: Feature data for prediction
            return_proba: If True and model is a classifier, return probability estimates

        Returns:
            Predictions or probability estimates
        """
        # SCALE
        if self.use_scaler == True:
            X = self.scaler.transform(X)
        
        # PREDICT
        if self.task == "classification" and return_proba == True:
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        # PREDICT
        y_pred = self.predict(X_test)

        if self.task == "classification":
            # need for the roc_auc_score
            y_proba = self.predict(X_test, return_proba = True)

            # if multi-class, then roc_auc_score is calculated differently
            n_classes = len(np.unique(y_test))

            if n_classes == 2:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # get just the positive class probability
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                precision = precision_score(y_test, y_pred, average = "weighted")
                recall = recall_score(y_test, y_pred, average = "weighted")
                f1 = f1_score(y_test, y_pred, average = "weighted")

                # ovr = one-vs-rest for mult-class
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
                roc_auc = roc_auc_score(y_test, y_proba, multi_class = "ovr")

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        else:
            # regression metrics
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)), 
                "mae": mean_absolute_error(y_test, y_pred), 
                "r2": r2_score(y_test, y_pred)
            }

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation results using sklearn cross_val_score
        """
        # TODO: Use Pipeline when scaling, and choose classifier/regressor based on task
        # TODO: Choose scoring metrics based on classification vs regression

        if self.task == "classification":
            model = GradientBoostingClassifier(**self.params)
            scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_ovr"]
        else:
            model = GradientBoostingRegressor(**self.params)
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

        if self.use_scaler == True:
            pipeline = Pipeline([("scaler", self.scaler), ("model", model)])
        else:
            pipeline = Pipeline([("model", model)])

        # TODO: Get mean, stdev of cross_val_score for each metric

        # had to use cross_validate instead of cross_val_score b/c cross_val_score doesn't support multi-class classification
        # couldn't put in all the metrics at once and it was computationally expensive to do in loop
        # n_jobs = -1 speeds up to 17 mins instead of 120 mins :)
        cv_results = cross_validate(pipeline, X, y, cv = cv, scoring = scoring, n_jobs = -1)
        results = {}
        for metric in scoring:
            results[metric] = {
                "mean": np.mean(cv_results[f'test_{metric}']),
                "std": np.std(cv_results[f'test_{metric}'])
            }

        return results

    def get_feature_importance(
        self, plot: bool = False, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importances

        Returns:
            DataFrame with feature importances
        """

        # TODO: Optionally plot a bar chart of top_n feature importances

        # had to use named_steps to get the feature importances for the scaled model
        if hasattr(self.model, "named_steps"):
            model = self.model.named_steps["model"]
        else:
            model = self.model
        
        fi = model.feature_importances_
        
        fi_df = pd.DataFrame({"feature": self.feature_names, "importance": fi })
        # sorts by importance in descending order
        fi_df = fi_df.sort_values(by = "importance", ascending = False).reset_index(drop = True)

        if plot == True:
            sns.barplot(x = "importance", y = "feature", data = fi_df.head(top_n), color = "#B88FFA")
            plt.title(f"Top {top_n} Feature Importances")
            plt.show()

        return fi_df

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "roc_auc_ovr",
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning

        Args:
            X: Feature data
            y: Target data
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric to evaluate

        Returns:
            Dictionary with best parameters and results
        """
        # TODO: Choose classifier or regressor based on task

        model = None
        if self.task == "classification":
            model = GradientBoostingClassifier(**self.params)
        else:
            model = GradientBoostingRegressor(**self.params)

        # TODO: Initialize GridSearchCV
        if self.use_scaler == True:
            pipeline = Pipeline([("scaler", self.scaler), ("model", model)])

            # stopping weird naming error
            new_grid = {f"model__{k}": v for k, v in param_grid.items()}

            GSCV = GridSearchCV(pipeline, param_grid = new_grid, cv = cv, scoring = scoring, n_jobs = -1)
        else:
            GSCV = GridSearchCV(model, param_grid, cv = cv, scoring = scoring, n_jobs = -1)

        # TODO: Perform grid search for hyperparameter tuning
        GSCV.fit(X, y)

        self.model = GSCV.best_estimator_
        best_params = GSCV.best_params_

        # remove the model__ prefix from the best params
        clean_params = {k.replace("model__", ""): v for k, v in best_params.items()}

        results = {
            "best_params": clean_params,
            "best_score": GSCV.best_score_
        }

        prefix = "param_model__" if self.use_scaler else "param_"

        cv_results = pd.DataFrame(GSCV.cv_results_)

        cv_df = cv_results[[f"{prefix}learning_rate",
                            f"{prefix}max_depth",
                            f"{prefix}n_estimators",
                            "mean_test_score",
                            "std_test_score"]]

        cv_df.rename(columns = {
            f"{prefix}learning_rate": "Learning Rate",
            f"{prefix}max_depth": "Max Depth",
            f"{prefix}n_estimators": "Number of Estimators",
            "mean_test_score": "Mean CV Score",
            "std_test_score": "Std CV Score"
        }, inplace = True)

        cv_df = cv_df.sort_values(by = "Mean CV Score", ascending = False)
        print()
        print(cv_df.head(5))
        print()

        return results

    def plot_tree(
        self, tree_index: int = 0, figsize: Tuple[int, int] = (20, 15)
    ) -> None:
        """
        Plot a specific tree from the ensemble

        Args:
            tree_index: Index of the tree to plot
            figsize: Figure size for the plot
        """

        # CITATION: CSCI1851class code for TreesAndForests

        plt.figure(figsize = figsize)

        # the model becomes a pipeline if the scaler is used so it doesn't have .estimators_
        if hasattr(self.model, "named_steps"):
            model = self.model.named_steps["model"]
        else:
            model = self.model

        # take first tree in ensemble (via EdStem)
        tree = model.estimators_[tree_index][0]

        plot_tree(
            tree,
            feature_names = self.feature_names,
            filled = True,
            rounded = True,
            max_depth = self.params["max_depth"], 
        )

        plt.title(f"Tree {tree_index}")
        plt.show()

