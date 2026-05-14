"""Unit-тести скрипта тренування."""
from pathlib import Path

import joblib

from ml.train import trainAndSave


def test_train_creates_model_file(tmp_path: Path):
    """Після виклику trainAndSave файл моделі має існувати, accuracy — у допустимих межах."""
    modelFile = tmp_path / "model.joblib"
    accuracy = trainAndSave(modelPath=modelFile)
    assert modelFile.exists(), "Файл моделі має бути створений"
    assert 0.0 <= accuracy <= 1.0, "Accuracy має бути коректним числом"
    assert accuracy > 0.8, f"Очікувано accuracy > 0.8, отримано {accuracy}"


def test_model_predicts_three_classes(tmp_path: Path):
    """Передбачення на типовому зразку класу_0 повертає валідний ідентифікатор класу."""
    modelFile = tmp_path / "model.joblib"
    trainAndSave(modelPath=modelFile)
    model = joblib.load(modelFile)
    # Типовий зразок класу_0 з датасету Wine: 13 ознак у тому ж порядку, що й feature_names
    sample = [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]
    pred = model.predict(sample)
    assert pred[0] in (0, 1, 2), "Клас має бути одним із 0/1/2"
