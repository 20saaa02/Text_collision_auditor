from dataclasses import dataclass
from typing import List


@dataclass
class ErrorClass:
    isError: bool
    messageError: str

@dataclass
class cleanTextGet:
    text: str

@dataclass
class cleanTextResult:
    text: str

@dataclass
class splittingTextIntoChunksGet:
    text: str
    chunk_size: int = 512  # Размер чанка в символах
    overlap: int = 128     # Перекрытие между чанками
    by_sentences: bool = True  # Делить по предложениям (True) или фиксированной длины (False)


@dataclass
class splittingTextIntoChunksResult:
    chunks: List[str]
    error: ErrorClass

