from os import PathLike
from typing import Any, Dict, List, Literal, NewType, Type, TypeVar, Union

from caseless_dictionary import CaselessDict

__all__ = [
    "StrOrListOf",
    "StrOrListDictOf",
    "StrNum",
    "StrNumOrListDictOf",
    "FileNameMatchSelector",
    "AnyFile",
    "AnyDir",
    "PathType",
    "PathOrStr",
    "AnyDict",
    "Definition",
]


StrOrListOf = TypeVar("StrOrListOf", str, List[str])
StrOrListDictOf = TypeVar("StrListorStrDict", str, List[str], Dict[str, Any])
StrNum = TypeVar("StrNum", str, float, int)
StrNumOrListDictOf = TypeVar(
    "StrNumOrListDictOf", StrNum, List[StrNum], Dict[StrNum, Any]
)
FileNameMatchSelector = TypeVar(
    "FileNameMatchSelector",
    Literal["full"],
    Literal["stem"],
    Literal["ext"],
    Literal["suffix"],
    Literal["parts"],
)


AnyFile = TypeVar("AnyFile", bound=Type["File"])
AnyDir = TypeVar("AnyDir", bound=Type["Dir"])
PathType = TypeVar("PathType", PathLike, AnyFile, AnyDir)

PathOrStr = TypeVar("PathOrStr", PathType, str)

AnyDict = TypeVar("AnyDict", Dict, CaselessDict)

Definition = NewType("Definition", Type["Definition"])
