from __future__ import annotations

from collections.abc import Callable

import pyparsing  # type: ignore

DataType = str
TreeList = str | list["TreeList"]
TreeTuple = tuple[DataType, tuple["TreeTuple", ...] | None]


class Tree:
    """
    A class for representing a generic tree structure

    Parameters
    ----------
    data : DataType
    children : list(Tree)

    Attributes
    ----------
    data : DataType
    children : list(Tree)
    terminals : list(str)

    Methods
    -------
    to_tuple, to_string,
    index, relabel,
    from_string, from_list,
    check_depth, check_for_loops
    """

    LPAR = pyparsing.Suppress("(")
    RPAR = pyparsing.Suppress(")")
    DATA = pyparsing.Regex(r"[^\(\)\s]+")

    PARSER = pyparsing.Forward()
    SUBTREE = pyparsing.ZeroOrMore(PARSER)
    PARSERLIST = pyparsing.Group(LPAR + DATA + SUBTREE + RPAR)
    PARSER <<= DATA | PARSERLIST

    def __init__(self, data: DataType, children: list["Tree"] | None = None):
        self._data = data
        self._children = children if children is not None else []

        self._validate()

    def to_tuple(self) -> TreeTuple:
        """
        Converts the tree into a tuple representation

        Returns
        -------
        TreeTuple
        """
        return self._data, tuple(c.to_tuple() for c in self._children)

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()

    def __str__(self) -> str:
        return " ".join(self.terminals)

    def __repr__(self) -> str:
        return self.to_string()

    def to_string(self, depth=0) -> str:
        """
        Returns a formatted string representation of the tree

        Parameters
        ----------
        depth : int

        Returns
        -------
        str
        """
        s = (depth - 1) * "  " + int(depth > 0) * "--" + self._data + "\n"
        s += "".join(c.to_string(depth + 1) for c in self._children)

        return s

    def __contains__(self, data: DataType) -> bool:
        if self._data == data:
            return True
        else:
            for child in self._children:
                if data in child:
                    return True

            return False

    def __getitem__(self, idx: int | tuple[int, ...]) -> Tree:
        if isinstance(idx, int):
            return self._children[idx]
        elif len(idx) == 1:
            return self._children[idx[0]]
        elif idx:
            return self._children[idx[0]].__getitem__(idx[1:])
        else:
            return self

    @property
    def data(self) -> DataType:
        return self._data

    @property
    def children(self) -> list[Tree]:
        return self._children

    @property
    def terminals(self) -> list[str]:
        if self._children:
            return [w for c in self._children for w in c.terminals]
        else:
            return [str(self._data)]

    def _validate(self) -> None:
        try:
            assert all(isinstance(c, Tree) for c in self._children)
        except AssertionError:
            msg = "all children must be trees"
            raise TypeError(msg)

    def index(
        self, data: DataType, index_path: tuple[int, ...] = tuple()
    ) -> list[tuple[int, ...]]:
        """
        Returns all indexed paths of the data

        Parameters
        ----------
        data : DataType
        index_path : tuple

        Returns
        -------
        list[tuple]
        """
        indices = [index_path] if self._data == data else []

        indices += [
            j
            for i, c in enumerate(self._children)
            for j in c.index(data, index_path + (i,))
        ]

        return indices

    def relabel(
        self,
        label_map: Callable[[DataType], DataType],
        nonterminals_only: bool = False,
        terminals_only: bool = False,
    ) -> Tree:
        """
        Relabels the tree's nodes based on the provided function, params can be set to
        optionally filter to non-terminals or terminals only

        Parameters
        ----------
        label_map : Callable[[DataType], DataType]
        nonterminals_only : bool
        terminals_only : bool

        Returns
        -------
        Tree
        """
        if not nonterminals_only and not terminals_only:
            data = label_map(self._data)
        elif nonterminals_only and self._children:
            data = label_map(self._data)
        elif terminals_only and not self._children:
            data = label_map(self._data)
        else:
            data = self._data

        children = [
            c.relabel(label_map, nonterminals_only, terminals_only)
            for c in self._children
        ]

        return self.__class__(data, children)

    @classmethod
    def from_string(cls, treestr: str) -> "Tree":
        """
        Creates a tree from a string representation

        Parameters
        ---------
        treestr : str

        Returns
        ------
        Tree
        """
        treelist = cls.PARSER.parseString(treestr[2:-2])[0]

        return cls.from_list(treelist)

    @classmethod
    def from_list(cls, treelist: TreeList) -> Tree:
        """
        Creates a tree from a nested list representation as produced by pyparsing.

        Parameters
        ---------
        treelist : TreeList
        A TreeList is either:
        - a str (becomes a leaf node)
        - a list where element 0 is the label and elements 1: are children

        Returns
        ------
        Tree
        """
        if isinstance(treelist, str):
            return cls(treelist)
        return cls(treelist[0], [cls.from_list(child) for child in treelist[1:]])

    def check_depth(self, depth: int = 0, threshold: int = 100) -> None:
        """
        Checks if the depth of the tree exceeds a given threshold

        Parameters
        ----------
        depth : int
        threshold : int

        Returns
        -------
        None

        """
        if depth > threshold:
            print(f"Warning: Tree depth exceeds expected range at depth {depth}")
        for child in self.children:
            child.check_depth(depth + 1)

    def check_for_loops(self, visited: set | None = None) -> None:
        """
        Checks if the tree has a loop that will cause some functions to hang.

        Parameters
        ----------
        visited : set(int)

        Returns
        -------
        None

        """
        if visited is None:
            visited = set()
        if id(self) in visited:
            raise Exception("Circular reference detected")
        visited.add(id(self))
        for child in self.children:
            child.check_for_loops(visited)
