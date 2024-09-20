from unittest.mock import MagicMock

import numpy as np
import pytest
from src.inference_model import XGBoostModel
from src.utils import clean_text


class MockInferenceModel(XGBoostModel):
    def __init__(self,
                 data_path: str,
                 target: str,
                 save_path: str):
        self.vectorizer = MagicMock(name="vectorizer")
        self.label_encoder = MagicMock(name="label_encoder")
        self.label_encoder.inverse_transform.return_value = "A"
        self.label_encoder.classes_ = ["A", "B"]
        self.model = MagicMock(name="model")
        self.model.predict.return_value = [0]
        self.model_name = "XGBClassifier"

        self.target = target
        self.data_path = data_path
        self.save_path = save_path


def test_clean_text(example_text_for_nlp, example_cleaned_text_for_nlp):
    cleaned_text = clean_text(example_text_for_nlp)
    assert cleaned_text == example_cleaned_text_for_nlp

# class TestInferenceModel:
#     model = MockInferenceModel(data_path="data", target="target", save_path="artifacts")
