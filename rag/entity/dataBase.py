from dataclasses import dataclass


@dataclass
class ErrorClass:
    isError: bool
    messageError: str