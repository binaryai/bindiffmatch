import dataclasses
from typing import Optional


@dataclasses.dataclass(slots=True, kw_only=True)
class BasicInfo:
    base_address: int
    file_type: Optional[str] = None
    machine_type: Optional[str] = None
    platform_type: Optional[str] = None
    endianness: Optional[str] = None
    loader: Optional[str] = None
    entrypoint: Optional[int] = None


@dataclasses.dataclass(slots=True, kw_only=True)
class Function:
    addr: int
    name: Optional[str] = None
    pseudocode: Optional[str] = None
    callees: Optional[list[int]] = None
    strings: Optional[list[str]] = None
    embedding: Optional[list[float]] = None
    linecount: Optional[int] = None


@dataclasses.dataclass(slots=True, kw_only=True)
class BinaryFile:
    sha256: str
    basic_info: BasicInfo
    functions: Optional[list[Function]] = None


# --------------------------------------


@dataclasses.dataclass(slots=True, kw_only=True)
class AlgorithmOutputMatchPair:
    addr_1: int
    addr_2: int
    score: Optional[float] = None


@dataclasses.dataclass(slots=True, kw_only=True)
class MatchPair:
    function_1: Function
    function_2: Function
    score: Optional[float] = None


@dataclasses.dataclass(slots=True, kw_only=True)
class MatchResult:
    file_1: BinaryFile
    file_2: BinaryFile
    matches: list[MatchPair]
    unmatches_1: list[Function]
    unmatches_2: list[Function]
