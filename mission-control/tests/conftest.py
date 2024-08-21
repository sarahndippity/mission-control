import pandas as pd
import pytest


@pytest.fixture
def example_imbalanced_dataset():
    df = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "feat": [1, 2, 2, 3, 1, 3, 3, 2, 1, 1],
        "class": ["a", "a", "b", "b", "b", "b", "a", "a", "a", "a"]
    })
    return df


@pytest.fixture
def mock_train_set():
    df = pd.DataFrame({
        "id": [0, 1, 4, 5, 6, 7],
        "feat": [1, 2, 1, 3, 3, 2],
        "class": ["a", "a", "b", "b", "a", "a"]
    })
    return df


@pytest.fixture
def mock_test_set():
    df = pd.DataFrame({
        "id": [2, 3, 8, 9],
        "feat": [2, 3, 1, 1],
        "class": ["b", "b", "a", "a"]
    })
    return df


@pytest.fixture
def example_multiclass_imbalanced_dataset():
    df = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "feat": [1, 2, 2, 3, 1, 3, 3, 2, 1, 1, 3, 3, 2, 1, 1],
        "class": ["a", "a", "b", "b", "b", "b", "a", "a", "a", "a", "c", "c", "c", "c", "c"]
    })
    return df
