from typing import Any, AnyStr, Dict, List, Literal, Union

StrOrListOfStr = Union[AnyStr, List[AnyStr]]
StrListOrStrDict = Union[str, List[str], Dict[str, Any]]
FileNameMatchSelector = Union[
    Literal["full", "stem", "ext", "suffix", "parts"]
]
GROFileType = "GROFile"
TOPFileType = "TOPFile"
ITPFileType = "ITPFile"
ITPFileType = "YAMLFile"
AnyFileType = Union[
    "GROFile", "ITPFile", "TOPFile", "File", "MDPFile", "YAMLFile"
]
AnyDirType = Union["Dir", "FFDir"]
BasicPathType = "BasicPath"
FileOrStr = Union[AnyFileType, BasicPathType]
FileListType = List[AnyFileType]
AnyPathType = Union[AnyFileType, BasicPathType, AnyDirType, "Path"]
