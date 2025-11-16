import numpy as np

from src.data_loader import load_iris_data
from src.model import IrisClassifier


def test_load_data():
    data = load_iris_data()
    assert data is not None
    assert len(data) > 0


def test_model_training():
    model = IrisClassifier()
    model.train([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]], [0, 2])
    assert model.is_trained
    assert model is not None


def test_model_prediction():
    model = IrisClassifier()
    model.train([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]], [0, 2])
    predictions = model.predict([[5.0, 3.6, 1.4, 0.2], [6.5, 3.0, 5.2, 2.0]])
    assert len(predictions) == 2
    assert all(
        isinstance(pred, (np.int32, np.int64, int)) for pred in predictions
        )
