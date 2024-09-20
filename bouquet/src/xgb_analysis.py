import joblib
import json
import logging
import os
import re
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score,
                             auc,
                             classification_report,
                             confusion_matrix,
                             f1_score,
                             multilabel_confusion_matrix,
                             precision_recall_curve,
                             roc_curve,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from utils import clean_text, plot_confusion_matrix

logger = logging.getLogger("xgb_analysis")

# TODO: Create version of XGBAnalyzer for non-NLP use cases

class XGBAnalyzer:
    def __init__(self,
                 artifact_path: str,
                 data_path: str,
                 target_variable: str,
                 random_seed: int = 13):
        """
        Class that contains various methods for analyzing results of XGBoost classification
        model.

        Parameters
        ----------
        artifact_path: str
            Local path to XGBoost model artifacts - vectorizer, label encoder, and model
        data_path: str
            Local path to training and test datasets
        target_variable: str
            Name of target variable
        random_seed: int, default 13
            Random seed
        """
        self.random_seed = random_seed

        self.vectorizer = joblib.load(f"{artifact_path}/vectorizer.pkl")
        self.label_encoder = joblib.load(f"{artifact_path}/label_encoder.pkl")
        self.model = XGBClassifier()
        self.model.load_model(f"{artifact_path}/model.json")
        self.target_variable = target_variable

        self.train = pd.read_csv(f"{data_path}/train.csv")
        self.test = pd.read_csv(f"{data_path}/test.csv")
        self.train_vectorized = ""
        self.test_vectorized = ""

    def calculate_metadata(self):
        """
        Calculates certain parameters/metadata necessary for downstream evaluation and
        analysis, including vectorized training data (text) and predictions and
        predicted probabilities.
        """
        logger.info("Cleaning and vectorizing text for train and test sets.")
        self.train["lemm_text"] = self.train["ocr_text"].apply(clean_text)
        self.train[self.target_variable] = self.train[self.target_variable].astype(str).str.upper()
        self.test["lemm_text"] = self.test["ocr_text"].apply(clean_text)
        self.test[self.target_variable] = self.test[self.target_variable].astype(str).str.upper()
        self.train_vectorized = self.vectorizer.transform(self.train["lemm_text"])
        self.test_vectorized = self.vectorizer.transform(self.test["lemm_text"])

        logger.info("Generating predictions on the test set.")
        y_pred = self.model.predict(self.test_vectorized)

        self.test[f"{self.target_variable}_pred"] = self.label_encoder.inverse_transform(y_pred)
        self.test[f"{self.target_variable}_prob"] = self.model.predict_proba(self.test_vectorized).max(axis=1)
        self.test[f"{self.target_variable}_correct"] = self.test[f"{self.target_variable}_pred"] == self.test[self.target_variable]

    def generate_classification_report(self):
        """
        Generate classification report.

        Returns
        -------
        dict
            Classification report as a dictionary
        """
        y_pred = self.test[f"{self.target_variable}_pred"]
        y_true = self.test[self.target_variable]
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))

        return report

    def plot_training_curves(self, show_plot=True):
        """
        Plots the training ROC curves. Aids in calculating the optimal classification
        threshold using Youden's J statistic.

        Parameters
        ----------
        show_plot: bool, default True
            If True, shows the ROC and PR plots.

        Returns
        -------
        Tuple(float, float, float)
            Returns 3 floats. The first is the optimal threshold value, the second is
            Youden's J statistic at the optimal threshold, and the third is the F1 score
            at the optimal threshold.
        """
        y_true = self.label_encoder.transform(self.test[self.target_variable])
        y_pred = self.label_encoder.transform(self.test[f"{self.target_variable}_pred"])

        # step 1: compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        # step 2: compute Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        # step 3: compute precision-recall curve and AUC
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        # calculate F1 score for the optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_f1 = f1_scores[np.argmax(pr_thresholds >= optimal_threshold)]

        # step 4: plot ROC and PR curves
        if show_plot:
            plt.figure(figsize=(14, 6))

            # ROC curve
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1],  color="gray", linestyle="--")
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color="red",
                        label=f"Optimal Threshold = {optimal_threshold:.2f}")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")

            # precision-recall curve
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, color="blue", label=f"PR curve (AUC = {pr_auc:.2f})")
            plt.scatter(recall[np.argmax(pr_thresholds >= optimal_threshold)],
                        precision[np.argmax(pr_thresholds >= optimal_threshold)],
                        color="red", label=f"Optimal Threshold (F1 = {optimal_f1:.2f})")
            plt.title("Precision-Recall (PR) Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower left")

            plt.tight_layout()
            plt.show()

        return optimal_threshold, j_scores[optimal_idx], optimal_f1

    def generate_confusion_matrices(self,
                                    figsize: Tuple[int, int] = (15, 6),
                                    fontsize: int = 10):
        """
        Generates visuals for confusion matrices.

        Parameters
        ----------
        figsize: tuple[int, int], default (15, 6)
            Tuple representing the figure size (length, width).
        fontsize: int, default 10
            Integer representing plot font size.
        """
        df = self.test.copy()
        class_names = df[self.target_variable].unique().tolist()
        cm = confusion_matrix(df[self.target_variable],
                              df[f"{self.target_variable}_pred"],
                              labels=class_names)

        plot_confusion_matrix(confusion_matrix=cm,
                              class_names=class_names,
                              errors_only=False,
                              figsize=figsize,
                              fontsize=fontsize)

    def plot_feature_importance(self,
                                num_features: int = 75,
                                figsize: Tuple[int, int] = (8, 8),
                                fontsize: int = 8):
        """
        Plot XGBoost feature importances.

        Parameters
        ----------
        num_features: int, default 75
            Number of features to plot. Takes the first N when sorted in descending order
            by importance.
        figsize: tuple[int, int], default (8, 8)
            Tuple representing the figure size (length, width).
        fontsize: int, default 8
            Integer representing plot font size.
        """
        feature_importance = self.model.feature_importances_
        feature_names = self.vectorizer.get_feature_names_out()
        sorted_idx = np.argsort(feature_importance)[-num_features:]

        fig = plt.figure(figsize=figsize)
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx], fontsize=fontsize)
        plt.title("Feature Importance")
        plt.show()

    def xgboost_feature_selection(self,
                                  max_features: int = 300,
                                  xgb_objective: str = "binary"):
        """
        Determine the optimal number of features for the XGBoost model. Attempts to
        retrain the model with incrementally fewer and fewer features and reports
        resulting performance.

        Parameters
        ----------
        max_features: int, default 300
            Maximum number of features to use in training the model. These would be the
            top N features as identified by model feature importance.
        xgb_objective: str, default 'binary'
            XGBoost model objective
        """
        y_train = self.label_encoder.transform(self.train[self.target_variable])
        y_test = self.label_encoder.transform(self.test[self.target_variable])

        # fit model using each importance as a threshold
        thresholds = np.sort(self.model.feature_importances_)[-(max_features + 1):]

        for i in range(0, max_features, 10):
            thresh = thresholds[i]
            # select features using threshold
            selection = SelectFromModel(self.model, threshold=thresh, prefit=True)
            select_X_train = selection.transform(self.train_vectorized)

            # train model
            selection_model = XGBClassifier(random_state=self.random_seed,
                                            objective=xgb_objective)
            selection_model.fit(select_X_train, y_train)

            # evaluate model
            select_X_test = selection.transform(self.test_vectorized)
            y_pred = selection_model.predict(select_X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            logger.info(f"Threshold {thresh}, n={select_X_train.shape[1]}, Accuracy: {accuracy}, F1: {f1}")

    def run_hyperparameter_optimization(self,
                                        parameter_distribution: dict,
                                        iterations: int = 30,
                                        cv: int = 3,
                                        n_jobs: int = 4,
                                        scoring: str = "f1_micro",
                                        save_model: bool = True) -> XGBClassifier:
        """
        Runs randomized grid search for hyperparameter optimization.
        https://www.kdnuggets.com/2022/08/tuning-xgboost-hyperparameters.html

        Parameters
        ----------
        parameter_distribution: dict
            Dictionary of all hyperparameter names and associated values to try optimizing
            with.
        iterations: int, default 30
            Maximum number of iterations to run.
        cv: int, default 3
            Number of folds to use in cross-validation.
        n_jobs: int, default 4
            Number of jobs to run.
        scoring: str, default 'f1_micro'
            Evaluation metric to measure performance by.
        save_model: bool, default True
            If True, saves best model (hyperparameter-optimized) locally.

        Returns
        -------
        XGBoost Classifier object
            Optimized XGBoost model
        """
        X_train = self.train_vectorized
        y_train = self.label_encoder.transform(self.train[self.target_variable])

        # create XGB model object
        xgb_model = XGBClassifier(random_state=self.random_seed)

        # create grid search object
        random_search = RandomizedSearchCV(xgb_model,
                                           param_distributions=parameter_distribution,
                                           n_iter=iterations,
                                           cv=cv,
                                           n_jobs=n_jobs,
                                           scoring=scoring,
                                           verbose=1)

        # fit the grid search object to training data
        random_search.fit(X_train, y_train)

        # print best set of hyperparameters
        print("Best set of hyperparameters: ", random_search.best_params_)
        print("Best score: ", random_search.best_score_)

        if save_model:
            save_path = os.path.join(self.artifact_path, "best_model.json")
            random_search.best_estimator_.save_model(save_path)
