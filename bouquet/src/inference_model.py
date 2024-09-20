import logging
import os
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from src.utils import clean_text

logger = logging.getLogger("inference_model")


class BaseNlpModel:
    def __init__(self):
        """
        Author: Sarah Xie
        Origin:

        Base model class for NLP classification projects with separate tokenization and
        prediction steps.
        Default base class objects include vectorized text and arrays of class labels.
        These are defined/generated properly later in concrete classes.
        """
        self.model_name = "BaseNlpModel"
        self.train = ""
        self.val = ""
        self.test = ""
        self.x_train_vectorized = ""
        self.x_test_vectorized = ""
        self.x_val_vectorized = ""
        self.y_train = ""
        self.y_test = ""
        self.y_val = ""

    def run(self):
        """
        Pulls all methods together and executes the training and evaluation of an NLP
        classifier.
        """
        self.load_data()
        self.tokenize()
        self.encode()
        self.train()
        self.predict()
        self.score()
        self.save_model()

    @abstractmethod
    def load_data(self):
        """
        Load data
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self):
        """
        Tokenize inputs
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self):
        """
        Encode labels
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
        Train classifier model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """
        Generate predictions
        """
        raise NotImplementedError

    @abstractmethod
    def score(self):
        """
        Evaluate model
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        """
        Save artifacts
        """
        raise NotImplementedError


class XGBoostModel(BaseNlpModel):
    def __init__(self,
                 data_path: str,
                 target: str,
                 max_features: int,
                 xgb_kwargs: dict,
                 save_path: str):
        """
        Class for training an XGBoost model to perform binary or multi-label classification
        on TF-IDF tokenized text.

        Parameters
        ----------
        data_path: str
            Path to location of data files (like train and test sets). Must be named
            "train", "test", etc. and must be CSVs.
        target: str
            Target variable for classification task. Must contain 2 or more labels.
        xgb_kwargs: dict
            Kwargs for XGBoost classifier
        max_features: int, default 10000
            Maximum number of features to be generated from TF-IDF vectorizer
        save_path: str
            Location for saving model artifacts
        """
        super().__init__()
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                                          norm="l2",
                                          ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.model = XGBClassifier(**xgb_kwargs) # works for both binary and multiclass problems as long as the proper kwargs are provided
        self.model_name = "XGBClassifier"

        self.target = target
        self.data_path = data_path
        self.save_path = save_path

    def load_data(self):
        """
        Loads data from the provided location.
        """
        self.train = pd.read_csv(f"{self.data_path}/train.csv")
        self.test = pd.read_csv(f"{self.data_path}/test.csv")
        try:
            self.val = pd.read_csv(f"{self.data_path}/val.csv")
        except FileNotFoundError:
            self.val = None

    def tokenize(self):
        """
        Cleans and tokenizes input text using TF-IDF.
        """
        start = time.time()
        logger.info("Cleaning text")
        self.train["cleaned_text"] = self.train["text"].apply(clean_text)
        self.test["cleaned_text"] = self.test["text"].apply(clean_text)
        if self.val:
            self.val["cleaned_text"] = self.val["text"].apply(clean_text)

        logger.info("Vectorizing text")
        self.x_train_vectorized = self.vectorizer.fit_transform(self.train["cleaned_text"])
        self.x_test_vectorized = self.vectorizer.transform(self.test["cleaned_text"])
        if self.val:
            self.x_val_vectorized = self.vectorizer.transform(self.val["cleaned_text"])
        end = time.time()
        logger.info(f"Text cleaned and vectorized in {end - start} seconds")

    def encode(self):
        """
        Encode labels for each dataset (train, test, val if relevant).
        """
        encoded_col = f"{self.target}_label"
        logger.info("Encoding class labels")
        self.train[encoded_col] = self.label_encoder.fit_transform(self.train[self.target])
        self.test[encoded_col] = self.label_encoder.transform(self.test[self.target])
        if self.val:
            self.val[encoded_col] = self.label_encoder.transform(self.val[self.target])

        self.y_train = self.train[encoded_col]
        self.y_test = self.test[encoded_col]
        if self.val:
            self.y_val = self.val[encoded_col]

    def train(self):
        """
        Train classifier.
        """
        logger.info(f"Fitting {self.model_name} for {self.target} classification")
        start = time.time()
        if self.val:
            self.model.fit(self.x_train_vectorized, self.y_train,
                           eval_set=[self.x_val_vectorized, self.y_val],
                           verbose=True)
        else:
            self.model.fit(self.x_train_vectorized, self.y_train,
                           verbose=True)
        end = time.time()
        logger.info(f"{self.model_name} trained in {end - start} seconds")

    def predict(self):
        """
        Generate predictions on test set using trained classifier to evaluate performance
        on yet unseen data.
        """
        logger.info("Generating predictions on test set")
        pred_col = f"{self.target}_pred"
        y_pred = self.model.predict(self.x_test_vectorized)
        self.test[pred_col] = self.label_encoder.inverse_transform(y_pred)

    def score(self) -> dict:
        """
        Generate a classification report demonstrating trained classifier's performance.

        Returns
        -------
        dict
            Classification report
        """
        report = classification_report(y_true=self.test[self.target],
                                       y_pred=self.test[f"{self.target}_pred"],
                                       output_dict=True)
        print(classification_report(y_true=self.test[self.target],
                                    y_pred=self.test[f"{self.target}_pred"]))

        return report

    def save_model(self):
        """
        Save model objects locally.
        """
        full_path = f"{self.save_path}/{self.model_name}"
        os.makedirs(full_path, exist_ok=True)
        joblib.dump(self.label_encoder, f"{full_path}/label_encoder.pkl")
        joblib.dump(self.vectorizer, f"{full_path}/vectorizer.pkl")

        self.model.save_model(f"{full_path}/xgb_model.json")
        logger.info(f"Model artifacts successfully saved to {full_path}")
