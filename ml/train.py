"""Тренування SVC-класифікатора на датасеті Wine.

Методичка пропонує Iris+LogisticRegression як приклад; для цього варіанта
використано Wine + SVC у Pipeline зі StandardScaler — стандартизація
обов'язкова, бо SVM чутливий до масштабу ознак.
"""
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"


def buildPipeline() -> Pipeline:
    """Збирає Pipeline: StandardScaler → SVC з probability=True.

    probability=True увімкнено, бо API повертає probability у відповіді
    /predict, а predict_proba у SVC доступне лише за цього прапора.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42)),
    ])


def trainAndSave(modelPath: Path = MODEL_PATH) -> float:
    """Навчає модель на Wine та повертає accuracy на тестовій вибірці."""
    X, y = load_wine(return_X_y=True)
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = buildPipeline()
    model.fit(xTrain, yTrain)
    accuracy = accuracy_score(yTest, model.predict(xTest))
    joblib.dump(model, modelPath)
    return accuracy


if __name__ == "__main__":
    acc = trainAndSave()
    print(f"Model trained. Test accuracy: {acc:.4f}")
    print(f"Saved to: {MODEL_PATH}")
