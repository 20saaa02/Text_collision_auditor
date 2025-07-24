from dataclasses import dataclass
from typing import List
import numpy as np
from entity.dataBase import ErrorClass

@dataclass
class textIntoFactsGet:
    text: str

@dataclass
class textIntoFactsResult:
    error: ErrorClass
    facts: List[str]

@dataclass
class prepareDBResult:
    error: ErrorClass

@dataclass
class prepareDBGet:
    text: str

@dataclass
class factsIntoEmbeddingsGet:
    sentences: list[str]

@dataclass
class factsIntoEmbeddingsResult:
    embeddings: np.ndarray
    error: ErrorClass

@dataclass
class checkCollisionOneGet:
    question: str

@dataclass
class checkCollisionOneResult:
    arrCollisions: List[str]
    error: ErrorClass

@dataclass
class checkCollisionAllGet:
    pass

@dataclass
class checkCollisionAllResult:
    arrCollisions: List[str]
    error: ErrorClass