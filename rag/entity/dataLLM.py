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

class CollisionAnswer(Enum):
    NO_COLLISION = auto()  # нейтральные / нет коллизии
    SUPPORT = auto()       # подтверждает
    COLLISION_IN_QUESTION = auto()
    COLLISION_BETWEEN_FACTS = auto()
    
    # строковое представление для удобства
    def __str__(self):
        return self.name.replace('_', ' ').title()

@dataclass
class findCollisionsResultOne:
    fact: str
    collisionType: CollisionAnswer

@dataclass
class findCollisionsResult:
    arrCollisionResult: List[findCollisionsResultOne]
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