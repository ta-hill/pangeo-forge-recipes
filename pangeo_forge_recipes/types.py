from dataclasses import dataclass
from enum import Enum
from typing import Dict, NamedTuple, Optional

from apache_beam.coders import Coder


class CombineOp(Enum):
    """Used to uniquely identify different combine operations across Pangeo Forge Recipes."""

    MERGE = 1
    CONCAT = 2
    SUBSET = 3


@dataclass(frozen=True, order=True)
class Dimension:
    """
    :param name: The name of the dimension we are combining over.
    :param operation: What type of combination this is (merge or concat)
    """

    name: str
    operation: CombineOp


@dataclass(frozen=True, order=True)
class Position:
    """
    :param indexed: If True, this position represents an offset within a dataset
       If False, it is a position within a sequence.
    """

    value: int
    # TODO: consider using a ClassVar here
    indexed: bool = False


@dataclass(frozen=True, order=True)
class IndexedPosition(Position):
    indexed: bool = True
    dimsize: int = 0


class Index(Dict[Dimension, Position]):
    """An Index is a special sort of dictionary which describes a position within
    a multidimensional set.

    - The key is a :class:`Dimension` which tells us which dimension we are addressing.
    - The value is a :class:`Position` which tells us where we are within that dimension.

    This object is hashable and deterministically serializable.
    """

    def __hash__(self):
        return hash(tuple(self.__getstate__()))

    def __getstate__(self):
        return sorted(self.items())

    def __setstate__(self, state):
        self.__init__({k: v for k, v in state})

    def find_concat_dim(self, dim_name: str) -> Optional[Dimension]:
        possible_concat_dims = [
            d for d in self if (d.name == dim_name and d.operation == CombineOp.CONCAT)
        ]
        if len(possible_concat_dims) > 1:
            raise ValueError(
                f"Found {len(possible_concat_dims)} concat dims named {dim_name} "
                f"in the index {self}."
            )
        elif len(possible_concat_dims) == 0:
            return None
        else:
            return possible_concat_dims[0]


class CodedGroupItem(NamedTuple):
    name: str
    val: int

    @classmethod
    def from_literal(cls, s_name, i_val):
        name = str(s_name.encode("utf-8"), encoding="utf-8").upper()
        val = int(bin(i_val), base=0)
        return cls(name, val)

    @classmethod
    def from_strings(cls, s_name, s_val):
        name = str(s_name.encode("utf-8"), encoding="utf-8").upper()
        val = int(bin(int(s_val)), base=0)
        return cls(name, val)


class CodedGroupItemCoder(Coder):
    def encode(self, group) -> bytes:
        return ("%s:%d" % (group.name, group.val)).encode("utf-8")

    func_encode = classmethod(encode)

    def decode(self, s):
        return CodedGroupItem.from_strings(*s.decode("utf-8").split(":"))

    def is_deterministic(self):
        return True

    def chain_encode(self, groups) -> bytes:
        return "&&".encode("utf-8").join(self.func_encode(g) for g in groups)

    func_chain_encode = classmethod(chain_encode)


CodedGroupKey = bytes  # Tuple[CodedGroupItem, ...]
