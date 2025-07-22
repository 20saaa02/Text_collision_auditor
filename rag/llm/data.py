from dataclasses import dataclass
from enum import Enum, auto
from typing import List


@dataclass
class ErrorClass:
    isError: bool
    messageError: str

@dataclass
class findTopNearestResult:
    topNearest: List[str]
    error: ErrorClass

@dataclass
class findTopNearestGet:
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