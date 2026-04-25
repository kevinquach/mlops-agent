from enum import Enum

from pydantic import BaseModel


class WineClass(str, Enum):
    CULTIVAR_1 = "Class 0 (Cultivar 1)"
    CULTIVAR_2 = "Class 1 (Cultivar 2)"
    CULTIVAR_3 = "Class 2 (Cultivar 3)"

    @classmethod
    def from_prediction(cls, prediction: int) -> "WineClass":
        mapping = {0: cls.CULTIVAR_1, 1: cls.CULTIVAR_2, 2: cls.CULTIVAR_3}
        return mapping[prediction]


class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class PredictionResponse(BaseModel):
    prediction: int
    wine_class: WineClass
    model_version: str


class FeatureLookupResponse(BaseModel):
    wine_id: int
    prediction: int
    wine_class: WineClass
    feature_source: str

FEATURE_COLS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]