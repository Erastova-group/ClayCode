from typing import NewType, Type, TypeVar, Union

__all__ = ["DefaultGMXRunType", "NondefaultGMXRun", "GMXRunType"]

DefaultGMXRunType = Type["GMXRun"]
NondefaultGMXRun = TypeVar(
    "NondefaultGMXRun",
    Type["EMRun"],
    Type["EQRun"],
    Type["EQRunFixed"],
    Type["EQRunRestrained"],
)
GMXRunType = TypeVar("GMXRunType", DefaultGMXRunType, NondefaultGMXRun)
