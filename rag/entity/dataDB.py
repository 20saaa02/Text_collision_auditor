from dataclasses import dataclass
from typing import List
import numpy as np
from entity.dataBase import ErrorClass


@dataclass
class initDBGet:
    db_path: str


@dataclass
class findTopNearestDBResult:
    topNearest: List[str]
    error: ErrorClass


@dataclass
class findTopNearestDBGet:
    query_embedding: np.ndarray
    k: int
    NofNearestCellsToCheck: int = 10


@dataclass
class loadEmbeddingsDBResult:
    error: ErrorClass


@dataclass
class loadEmbeddingsDBGet:
    sentence_embeddings: np.ndarray
    sentences: List[str]


@dataclass
class saveDBResult:
    error: ErrorClass


@dataclass
class saveDBGet:
    pass


@dataclass
class loadDBResult:
    error: ErrorClass


@dataclass
class loadDBGet:
    db_path: str = None
