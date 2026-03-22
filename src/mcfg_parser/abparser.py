from __future__ import annotations
from collections import defaultdict, deque
from abc import ABC

from .grammar import (
    MCFGRuleElement,
    MCFGRuleElementInstance,
    MultipleContextFreeGrammar,
)
from .tree import Tree
from .parser import Parser, NormalForm


ABBackPointer = tuple[int, MCFGRuleElement]


class ABEntry:
    """
    A chart entry for an agenda based parser chart

    Parameters
    ----------
    symbol : MCFGRuleElementInstance
    index : int
    backpointers : tuple[int, MCFGRuleElement]

    Attributes
    ----------
    symbol : MCFGRuleElementInstance
    index : int
    backpointers : tuple[int, MCFGRuleElement]
    """

    def __init__(
        self,
        symbol: MCFGRuleElementInstance,
        index: int,
        backpointers: tuple[ABBackPointer, ...] | None,
    ):
        self._symbol = symbol
        self._index = index
        self._backpointers = backpointers

    def to_tuple(self):
        return (self._symbol.variable, self.index, self._backpointers)

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other) -> bool:
        if isinstance(other, ABEntry):
            return self.to_tuple() == other.to_tuple()
        raise TypeError(
            f"Cannot compare {self.__class__.__name__} with {type(other).__name__}"
        )

    def __repr__(self) -> str:
        prefix = f"{self.index}:{self._symbol.variable}{self._symbol.string_spans}"
        if self._backpointers is None or all(bp is None for bp in self._backpointers):
            return f"{prefix} -> (None, None)"
        bp_left, bp_right = self._backpointers
        return f"{prefix} -> ({bp_left[0]}, {str(bp_left[1])}, {bp_right[0]}, {str(bp_right[1])})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def symbol(self) -> MCFGRuleElementInstance:
        return self._symbol

    @property
    def backpointers(self) -> tuple[ABBackPointer, ...] | None:
        return self._backpointers

    @property
    def index(self) -> int:
        return self._index


class AgendaBasedParser(Parser):
    """
    An agenda based parser

    Parameters
    ----------
    grammar : MultipleContextFreeGrammar
    """

    normal_form = NormalForm.CNF

    def __init__(self, grammar: MultipleContextFreeGrammar):
        self._grammar = grammar

    @property
    def grammar(self):
        return super().grammar

    def _parse(self, string: list[str]) -> set[Tree]:
        chart = self._fill_chart(string)
        sv = {ele.variable for ele in self.grammar._start_variables}
        start_nodes = [
            entry
            for entry in chart
            if entry.symbol.variable in sv
            and entry.symbol._string_spans == ((0, len(string)),)
        ]
        if len(start_nodes) >= 1:
            return {self._construct_parses(chart, string, s) for s in start_nodes}
        else:
            print("Unable to parse this sentence.")
        return set()

    def _recognize(self, string: list[str]) -> bool:
        chart = self._fill_chart(string)
        sv = {ele.variable for ele in self.grammar._start_variables}
        return any(
            [
                entry
                for entry in chart
                if entry.symbol.variable in sv
                and entry.symbol._string_spans == ((0, len(string)),)
            ]
        )

    def _combine(
        self, current: ABEntry, element: ABEntry
    ) -> tuple[int, tuple[MCFGRuleElementInstance] | None]:
        reversed = 0
        possible_rules = self.grammar.reduce(current.symbol, element.symbol)
        if possible_rules == set():
            reversed = 1
            possible_rules = self.grammar.reduce(element.symbol, current.symbol)
            if possible_rules == set():
                return 0, None
            else:
                result = tuple(
                    i
                    for i in [
                        r.instantiate_left_side(element.symbol, current.symbol)
                        for r in possible_rules
                    ]
                    if i is not None
                )
                return reversed, result
        else:
            result = tuple(
                i
                for i in [
                    r.instantiate_left_side(current.symbol, element.symbol)
                    for r in possible_rules
                ]
                if i is not None
            )
        return reversed, result

    def _fill_chart(self, string: list[str]) -> list[ABEntry]:
        if not string:
            return []

        agenda = deque()
        for idx, word in enumerate(string):
            possible_rules = self.grammar.parts_of_speech(word)
            for rule in possible_rules:
                agenda.append(
                    ABEntry(
                        rule.instantiate_left_side(
                            MCFGRuleElementInstance(word, (idx, idx + 1))
                        ),
                        0,
                        None,
                    )
                )

        next_id = 0
        for n, e in enumerate(agenda):
            e._index = n
            next_id = n + 1

        chart = []
        chart_ids = set()
        chart_by_var = defaultdict(list)

        while agenda:
            current = agenda.popleft()
            candidate_vars = self.grammar.get_partner_vars(current.symbol.variable)
            candidates = [
                entry for var in candidate_vars for entry in chart_by_var.get(var, [])
            ]
            for element in candidates:
                r, combination = self._combine(current, element)
                if combination:
                    for c in combination:
                        if r:
                            new_parse = ABEntry(
                                c,
                                next_id,
                                (
                                    (element.index, element.symbol.variable),
                                    (current.index, current.symbol.variable),
                                ),
                            )
                        else:
                            new_parse = ABEntry(
                                c,
                                next_id,
                                (
                                    (current.index, current.symbol.variable),
                                    (element.index, element.symbol.variable),
                                ),
                            )
                        next_id += 1
                        agenda.append(new_parse)

            if current.index not in chart_ids:
                chart.append(current)
                chart_ids.add(current.index)
                chart_by_var[current.symbol.variable].append(current)

        return chart

    def _get_item(self, inventory: list[ABEntry], index: int) -> ABEntry | None:
        if inventory:
            for i in inventory:
                if i.index == index:
                    return i
        return None

    def _construct_parses(
        self, chart: list[ABEntry], string: list[str], entry: ABEntry | None = None
    ) -> Tree:
        if entry is None:
            raise ValueError(
                "Cannot construct parse tree: chart entry not found (likely a malformed chart)"
            )
        if not entry.backpointers or not any(
            bp for bps in entry.backpointers for bp in bps
        ):
            span = entry.symbol.string_spans[0]
            terminal = " ".join(string[span[0] : span[1]])
            return Tree("".join([entry.symbol.variable, "(", terminal, ")"]))
        children = []
        for child_entry_id, _ in entry.backpointers:
            child_entry = self._get_item(chart, child_entry_id)
            children.append(self._construct_parses(chart, string, child_entry))
        return Tree(entry.symbol.variable, children)
