import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from src.data_loader import get_feature_names, load_iris_data
from src.model import IrisClassifier


def test_load_iris_data_shapes():
    X_train, X_test, y_train, y_test = load_iris_data()
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert y_train.ndim == 1
    assert y_test.ndim == 1
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0


def test_feature_names_length():
    feature_names = get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) == 4


def test_model_training_and_prediction():
    X_train, X_test, y_train, y_test = load_iris_data()
    clf = IrisClassifier()
    clf.train(X_train, y_train)
    preds = clf.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert preds.min() >= 0
    assert preds.max() <= 2
