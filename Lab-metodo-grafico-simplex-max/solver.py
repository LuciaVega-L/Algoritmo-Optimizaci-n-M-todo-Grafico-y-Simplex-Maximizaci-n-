# solver.py
from abc import ABC, abstractmethod

class Solver(ABC):
    def __init__(self, coefObjetivo, restricciones):
        self.coefObjetivo = coefObjetivo
        self.restricciones = restricciones

    @abstractmethod
    def resolver(self):
        pass