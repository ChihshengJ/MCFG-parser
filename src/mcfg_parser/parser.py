from abc import ABC
from enum import Enum

from .tree import Tree


class NormalForm(Enum):
    CNF = 0
    BNF = 1
    GNF = 2


class Parser(ABC):
    """
    An general parser class

    Parameters
    ----------
    grammar
    """

    normal_form = NormalForm.CNF

    def __init__(self, grammar):
        self._grammar = grammar

    def __call__(self, string, mode="recognize"):
        if mode == "recognize":
            return self._recognize(string)
        elif mode == "parse":
            return self._parse(string)
        else:
            msg = 'mode must be "parse" or "recognize"'
            raise ValueError(msg)

    def _recognize(self, string: list[str]) -> bool:
        raise NotImplementedError

    def _parse(self, string: list[str]) -> set[Tree]:
        raise NotImplementedError

    @property
    def grammar(self):
        return self._grammar
