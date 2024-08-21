from src.dataset_balancer import DatasetBalancer


class TestDatasetBalancer:
    target_variable = "class"
    stratify_cols = ["class"]
    train_ratio = 0.7
    test_ratio = 0.3

    def test_train_test_split_no_val(self, example_imbalanced_dataset):
        tts = DatasetBalancer(data=example_imbalanced_dataset,
                              target_variable=self.target_variable,
                              stratify_cols=self.stratify_cols,
                              balance_strategy="both",
                              train_ratio=self.train_ratio,
                              test_ratio=self.test_ratio)
        tts.execute_train_test_split()

        assert tts.train.shape == (7, 3)
        assert tts.test.shape == (3, 3)
        assert tts.val.shape == (0, 0)

    def test_train_test_split_w_val(self, example_imbalanced_dataset):
        tts = DatasetBalancer(data=example_imbalanced_dataset,
                              target_variable=self.target_variable,
                              stratify_cols=self.stratify_cols,
                              balance_strategy="both",
                              train_ratio=0.5,
                              test_ratio=0.3)
        tts.execute_train_test_split()

        assert tts.train.shape == (5, 3)
        assert tts.test.shape == (3, 3)
        assert tts.val.shape == (2, 3)

    def test_oversampling(self, example_imbalanced_dataset, mock_train_set, mock_test_set):
        ovr = DatasetBalancer(data=example_imbalanced_dataset,
                              target_variable=self.target_variable,
                              stratify_cols=self.stratify_cols,
                              balance_strategy="oversample",
                              train_ratio=self.train_ratio,
                              test_ratio=self.test_ratio)
        ovr.train = mock_train_set
        ovr.test = mock_test_set

        ovr.execute_balance_datasets()
        class_counts = ovr.train.groupby("class").size().to_dict()

        assert ovr.train.shape == (8, 3)
        assert class_counts["a"] == 4
        assert class_counts["b"] == 4

    def test_undersampling(self, example_imbalanced_dataset, mock_train_set, mock_test_set):
        und = DatasetBalancer(data=example_imbalanced_dataset,
                              target_variable=self.target_variable,
                              stratify_cols=self.stratify_cols,
                              balance_strategy="undersample",
                              train_ratio=self.train_ratio,
                              test_ratio=self.test_ratio)
        und.train = mock_train_set
        und.test = mock_test_set

        und.execute_balance_datasets()
        class_counts = und.train.groupby("class").size().to_dict()

        assert und.train.shape == (4, 3)
        assert class_counts["a"] == 2
        assert class_counts["b"] == 2

    def test_multiclass_multisampling(self, example_multiclass_imbalanced_dataset):
        mms = DatasetBalancer(data=example_multiclass_imbalanced_dataset,
                              target_variable=self.target_variable,
                              stratify_cols=self.stratify_cols,
                              balance_strategy="both",
                              train_ratio=self.train_ratio,
                              test_ratio=self.test_ratio,
                              sampling_class="c")
        train, val, test = mms.execute_data_balancer()
        train_counts = train.groupby("class").size().to_dict()
        test_counts = test.groupby("class").size().to_dict()

        assert train.shape == (9, 3)
        assert train_counts["a"] == 3
        assert train_counts["b"] == 3
        assert train_counts["c"] == 3

        assert val.shape == (0, 0)

        assert test.shape == (5, 3)
        assert test_counts["a"] == 2
        assert test_counts["b"] == 1
        assert test_counts["c"] == 2

