import logging
from typing import List, Literal, Optional, Tuple

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')


class DatasetBalancer:
    def __init__(self,
                 data: pd.DataFrame,
                 target_variable: str,
                 stratify_cols: List[str],
                 balance_strategy: Literal["oversample", "undersample", "both"],
                 train_ratio: float,
                 test_ratio: float,
                 sampling_class: Optional[str] = None,
                 random_state: int = 13):
        """
        Author: Sarah Xie
        Origin: https://github.com/sarahndippity/sx-portfolio/mission-control
        
        Prepare an imbalanced dataset for supervised classification tasks. Splits the
        given dataset into training, validation, and test sets, and balances the
        training and validation sets across imbalanced classes according to provided 
        parameters.
        Always balances the classes to have exact equal representation in the final set.
        Does not balance the test set.

        Parameters
        ----------
        data: pd.DataFrame
            Dataset to be used in training a classification model.
        target_variable: str
            Name of the column in dataset containing the target variable for classification
            task.
        stratify_cols: List[str]
            List of column names (strings) to use in stratifying the dataset when performing
            train/test split. This list should not be left empty - if no specific
            stratification is desired, use the target variable to stratify the dataseet.
            colu
        balance_strategy: str
            The balancing strategy to be used in selecting records according to their class
            in the target variable for training, validation, and test sets proportionally.
            Must be one of "oversample", "undersample", or "both".
        train_ratio: float
            The ratio of the original dataset to attribute to the training dataset.
            Used in conjunction with test_ratio to calculate the ratio to attribute to the
            validation set.
        test_ratio: float
            The ratio of the original dataset to attribute to the test dataset.
            Used in conjunction with train_ratio to calculate the ratio to attribute to the
            validation set.
        sampling_class: Optional[str], default None
            The name of the class in the target variable to use as an "anchor". Other classes
            will be over- or under-sampled depending on how many records they contain
            compared to this anchor class. When the sampling strategy is "both", user must
            provide a class name here. It will be None/unused otherwise.
        random_state: int, default 13
            Random state number to stabilize randomization for repeatability.

        Example
        -------
        >>> db = DatasetBalancer(data, "target", ["stratify1", "stratify2"],
        >>>                      "oversample", 0.6, 0.3)
        >>> train, val, test = db.execute_data_balancer()
        """
        self.random_state = random_state

        self.data = data
        self.target_variable = target_variable
        self.stratify_cols = stratify_cols
        self.balance_strategy = balance_strategy
        self.sampling_class = sampling_class
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = 1 - (self.train_ratio + self.test_ratio)

        self.train = pd.DataFrame()
        self.val = pd.DataFrame()
        self.test = pd.DataFrame()

    def execute_data_balancer(self):
        """
        Performs data balancing. Splits dataset into train, validation, and test sets
        and balances any imbalanced classes in the training & validation sets.

        Returns
        -------
        pd.DataFrame, pd.DataFrame, pd.DataFrame
            3 separate dataframes representing the training, validation, and test sets
            (in that order).
        """
        logger.info("Splitting data into train, val, and test sets.")
        self.execute_train_test_split()
        self.execute_balance_datasets()

        logger.info("Successfully generated balanced training, validation, and test datasets.")
        return self.train, self.val, self.test

    def execute_train_test_split(self):
        """
        Splits original dataset into smaller subsets of data useful for training,
        validation, and test/evaluation of a model. Uses the ratios given in DatasetBalancer
        instantiation to separate data.
        """
        df = self.data.copy()
        logger.info(f"Starting dataset size: {df.shape}")

        df["stratify_index"] = df[self.stratify_cols].apply(lambda x: "-".join([str(obj) for obj in x.values]),
                                                            axis=1)
        X = df.drop(labels="stratify_index", axis=1)
        y = df["stratify_index"]

        try:
            assert_almost_equal(self.val_ratio, 0.0, decimal=5), "No validation set"

            X_train, X_test, _y_train, _y_test = train_test_split(
                X, y, train_size=self.train_ratio, random_state=self.random_state, stratify=y
            )
            self.train = X_train.copy()
            self.test = X_test.copy()
            logger.info(f"Training data size - {self.train.shape}, Test data size - {self.test.shape}")
        except AssertionError:
            total_data_size = df.shape[0]
            val_slice = self.val_ratio * total_data_size
            leftover_slice = (1 - self.train_ratio) * total_data_size
            new_val_ratio = round(val_slice / leftover_slice, 2)

            X_train, X_rem, _y_train, y_rem = train_test_split(
                X, y, train_size=self.train_ratio, random_state=self.random_state, stratify=y
            )
            X_val, X_test, _y_val, _y_test = train_test_split(
                X_rem, y_rem,
                train_size=new_val_ratio,
                random_state=self.random_state,
                stratify=y_rem
            )
            self.train = X_train.copy()
            self.val = X_val.copy()
            self.test = X_test.copy()
            logger.info(f"Training data size - {self.train.shape}, Validation data size - {self.val.shape}, Test data size - {self.test.shape}")

    def execute_balance_datasets(self):
        """
        Balance datasets according to provided strategy (one of "oversample", "undersample",
        or "both" - oversampling and undersampling above/below the provided class in the
        target variable).
        """
        train_df = self.train.copy()
        val_df = self.val.copy()

        if train_df.empty:
            raise RuntimeError("Must provide non-empty training dataset. Try executing train/test split on original dataset before balancing.")

        if self.balance_strategy == "oversample":
            logger.info("Oversampling the minority class.")
            X_train, y_train = self._exec_oversampling(train_df)
            X_train[self.target_variable] = y_train
            self.train = X_train.copy()

            if not val_df.empty:
                X_val, y_val = self._exec_oversampling(val_df)
                X_val[self.target_variable] = y_val
                self.val = X_val.copy()

        elif self.balance_strategy == "undersample":
            logger.info("Undersampling the majority class.")
            X_train, y_train = self._exec_undersampling(train_df)
            X_train[self.target_variable] = y_train
            self.train = X_train.copy()

            if not val_df.empty:
                X_val, y_val = self._exec_undersampling(val_df)
                X_val[self.target_variable] = y_val
                self.val = X_val.copy()

        else:
            self.train = self._exec_both_over_and_undersampling(train_df)

            if not val_df.empty:
                self.val = self._exec_both_over_and_undersampling(val_df)

        logger.info(f"Dataset balancing complete. Final dataset size for train set: {self.train.shape}")
        if not val_df.empty:
            logger.info(f"validation set: {self.val.shape}")

    def _exec_oversampling(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
        """
        Balance a given dataset using oversampling technique.

        Parameters
        ----------
        dataset: pd.DataFrame
            Dataset to execute oversampling on, containing unique identifiers and the
            target variable with imbalanced classes in its own column.

        Returns
        -------
        Tuple(pd.DataFrame, np.array)
            The re-sampled dataset in two objects. A dataframe containing the re-sampled
            identifiers and other data, and an array containing the re-sampled target variable.
        """
        df = dataset.copy()
        y = df[self.target_variable]
        X = df.drop(labels=[self.target_variable], axis=1)

        ros = RandomOverSampler(random_state=self.random_state)
        X_ros, y_ros = ros.fit_resample(X, y)

        return X_ros, y_ros

    def _exec_undersampling(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
        """
        Balance a given dataset using undersampling technique.

        Parameters
        ----------
        dataset: pd.DataFrame
            Dataset to execute undersampling on, containing unique identifiers and the
            target variable with imbalanced classes in its own column.

        Returns
        -------
        Tuple(pd.DataFrame, np.array)
            The re-sampled dataset in two objects. A dataframe containing the re-sampled
            identifiers and other data, and an array containing the re-sampled target variable.
        """
        df = dataset.copy()
        y = df[self.target_variable]
        X = df.drop(labels=[self.target_variable], axis=1)

        rus = RandomUnderSampler(random_state=self.random_state)
        X_rus, y_rus = rus.fit_resample(X, y)

        return X_rus, y_rus

    def _exec_both_over_and_undersampling(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Balance a given dataset using both over and under-sampling techniques.

        Parameters
        ----------
        dataset: pd.DataFrame
            Dataset to execute undersampling on, containing unique identifiers and the
            target variable with imbalanced classes in its own column.

        Returns
        -------
        pd.DataFrame
            The re-sampled dataset including a column for the re-sampled target variable.
        """
        df = dataset.copy()

        # determine sampling class size
        class_counts = df.groupby(self.target_variable).size().to_dict()
        class_size = class_counts[self.sampling_class]
        under = []
        over = []
        for k, v in class_counts.items():
            if v <= class_size:
                over.append(k)
            if v >= class_size:
                under.append(k)

        # separate data for respective over/under-sampling
        under_df = df.loc[df[self.target_variable].isin(under)]
        over_df = df.loc[df[self.target_variable].isin(over)]
        var_df = df.loc[df[self.target_variable].isin([self.sampling_class])]

        logger.info(f"Undersampling the classes with majority representation above {self.sampling_class}")
        under_X_res, under_y_res = self._exec_undersampling(under_df)

        logger.info(f"Oversampling the classes with minority representation below {self.sampling_class}")
        over_X_res, over_y_res = self._exec_oversampling(over_df)

        under_X_res[self.target_variable] = under_y_res
        over_X_res[self.target_variable] = over_y_res

        # drop the anchor class from both over & under-sampled sets
        over_X_res = over_X_res.loc[over_X_res[self.target_variable] != self.sampling_class]
        under_X_res = under_X_res.loc[under_X_res[self.target_variable] != self.sampling_class]
        balanced_df = pd.concat([under_X_res, var_df, over_X_res], ignore_index=True)

        return balanced_df

