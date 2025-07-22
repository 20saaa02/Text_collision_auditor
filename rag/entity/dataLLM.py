from dataclasses import dataclass
from typing import List
from entity.dataBase import ErrorClass

@dataclass
class findTopNearestLLMResult:
    topNearest: List[str]
    error: ErrorClass

@dataclass
class findTopNearestLLMGet:
    question: str
    topFacts: List[str]
    countFind: int

@dataclass
class findCollisionsResult:
    arrCollisionResult: List[str]
    error: ErrorClass

@dataclass
class findCollisionsGet:
    question: str
    topFacts: List[str]


@dataclass
class splittingChunksIntoFactsResult:
    arrFacts: List[str]
    error: ErrorClass

@dataclass
class splittingChunksIntoFactsGet:
    chunk: str

@dataclass
class initLLMServiceGet:
    pass