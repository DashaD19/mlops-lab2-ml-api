"""Pydantic-моделі для контракту API.

Поля WineFeatures повторюють порядок і назви feature_names з
sklearn.datasets.load_wine() — це гарантує, що np.array з полів схеми
відповідає очікуванням навченого Pipeline.
"""
from pydantic import BaseModel, Field


class WineFeatures(BaseModel):
    """13 фізико-хімічних характеристик зразка вина."""

    alcohol: float = Field(..., ge=0, description="Вміст алкоголю, % об.")
    malic_acid: float = Field(..., ge=0, description="Яблучна кислота, г/л")
    ash: float = Field(..., ge=0, description="Зола, г/л")
    alcalinity_of_ash: float = Field(..., ge=0, description="Лужність золи")
    magnesium: float = Field(..., ge=0, description="Магній, мг/л")
    total_phenols: float = Field(..., ge=0, description="Загальні феноли")
    flavanoids: float = Field(..., ge=0, description="Флавоноїди")
    nonflavanoid_phenols: float = Field(..., ge=0, description="Не-флавоноїдні феноли")
    proanthocyanins: float = Field(..., ge=0, description="Проантоціани")
    color_intensity: float = Field(..., ge=0, description="Інтенсивність кольору")
    hue: float = Field(..., ge=0, description="Відтінок")
    od280_od315_of_diluted_wines: float = Field(..., ge=0, description="OD280/OD315 розведених вин")
    proline: float = Field(..., ge=0, description="Пролін, мг/л")


class PredictionResponse(BaseModel):
    """Відповідь /predict: ідентифікатор класу, його назва, ймовірність."""

    class_id: int
    class_name: str
    probability: float
