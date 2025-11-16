import pytest
from src.train import load_data, preprocess, train_model

def test_load_data():
    """Test that data loads correctly"""
    data = load_data()
    assert data is not None
    assert len(data) > 0

def test_preprocess_format():
    """Test that preprocessing returns correct format"""
    sample_data = [[1, 2, 3], [4, 5, 6]]
    result = preprocess(sample_data)
    assert isinstance(result, list)
    assert len(result) == 2

def test_model_training():
    """Test that model trains without errors"""
    # Create simple test data
    X_train = [[1, 2], [3, 4]]
    y_train = [0, 1]
    model = train_model(X_train, y_train)
    assert model is not None