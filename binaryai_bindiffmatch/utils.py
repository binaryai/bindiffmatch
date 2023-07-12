import copy
import dataclasses
import json
import os
import typing
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

from .models import AlgorithmOutputMatchPair, BasicInfo, BinaryFile, Function, MatchPair, MatchResult


def tranverse_path(src: str, dst: str) -> Generator[tuple[str, str], None, None]:
    try:
        f = open(src, "rb")
        f.close()
        yield (src, dst)
    except IsADirectoryError:
        for srcpath in Path(src).glob("**/*"):
            if not srcpath.is_file():
                continue
            rel_srcpath = srcpath.relative_to(src)
            dstpath = Path(dst).joinpath(rel_srcpath)
            yield (str(srcpath), str(dstpath))


def tranverse_path_two(src1: str, src2: str, dst: str) -> Generator[tuple[str, str, str], None, None]:
    try:
        f = open(src1, "rb")
        f.close()
        yield (src1, src2, dst)
    except IsADirectoryError:
        for src1path in Path(src1).glob("**/*"):
            if not src1path.is_file():
                continue
            rel_src1path = src1path.relative_to(src1)
            src2path = Path(src2).joinpath(rel_src1path)
            dstpath = Path(dst).joinpath(rel_src1path)
            yield (str(src1path), str(src2path), str(dstpath))


def unserialize(data: Any, objtype: Any, *, deepcopy: bool = True) -> Any:
    def _get_possible_type_of_optional(_objtype: Any) -> Any:
        if typing.get_origin(_objtype) is typing.Union:
            args = typing.get_args(_objtype)
            if len(args) == 2 and args[1] is type(None):  # noqa: E721
                return args[0]
        return None

    def _get_real_type(_objtype: type) -> Any:
        if (r := typing.get_origin(_objtype)) is not None:
            return r
        return _objtype

    def _get_matched(_t: Any, _declare: Any) -> Any:
        if _declare is Any or _t is _declare:
            return _declare
        elif (_origin := typing.get_origin(_declare)) is not None:
            if _origin is typing.Union:  # this implied typing.Optional
                for _union_arg in typing.get_args(_declare):
                    matched = _get_matched(_t, _union_arg)
                    if matched is not None:
                        return matched
                return None
            elif issubclass(_t, _origin):  # for typing.GenericAlias
                return _declare  # should return origin one
            else:
                return None
        elif issubclass(_t, _declare):  # for basic type
            return _declare
        else:
            return None

    def _get_list_type_arg(_objtype: Any) -> Any:
        _real = _get_real_type(_objtype)
        assert _real is Any or issubclass(_real, (list, tuple)), _real
        objtype_args = typing.get_args(_objtype)
        item_type = objtype_args[0] if len(objtype_args) >= 1 else Any
        return item_type

    def _get_dict_type_arg(_objtype: Any) -> tuple[Any, Any]:
        _real = _get_real_type(_objtype)
        assert _real is Any or issubclass(_real, dict), _real
        objtype_args = typing.get_args(_objtype)
        key_type = objtype_args[0] if len(objtype_args) >= 1 else Any
        val_type = objtype_args[1] if len(objtype_args) >= 2 else Any
        return key_type, val_type

    def _unserialize_internal(_data: Any, _objtype: Any, *, deepcopy: bool) -> Any:
        matched = _get_matched(type(_data), _objtype)
        if isinstance(_data, (list, tuple)):
            assert matched is not None
            item_type = _get_list_type_arg(matched)
            return type(_data)(_unserialize_internal(v, item_type, deepcopy=deepcopy) for v in _data)
        elif isinstance(_data, dict):
            if matched is not None:  # normal dict
                key_type, val_type = _get_dict_type_arg(matched)
                return type(_data)(
                    (
                        _unserialize_internal(k, key_type, deepcopy=deepcopy),
                        _unserialize_internal(v, val_type, deepcopy=deepcopy),
                    )
                    for k, v in _data.items()
                )
            else:  # maybe nesting dataclasses
                maybe_dataclass = _get_possible_type_of_optional(_objtype)
                if maybe_dataclass is None:
                    maybe_dataclass = _objtype
                field_types = {f.name: f.type for f in dataclasses.fields(maybe_dataclass)}
                fields = {}
                for k, v in _data.items():
                    if k not in field_types:
                        continue  # XXX: ignore extra items
                    assert isinstance(k, str)
                    fields[k] = _unserialize_internal(v, field_types[k], deepcopy=deepcopy)
                return maybe_dataclass(**fields)
        else:
            if not matched:
                return _objtype(_data)  # try to convert
            return copy.deepcopy(_data) if deepcopy else _data

    return _unserialize_internal(data, objtype, deepcopy=deepcopy)


def dict_factory_ignore_none(x: list[tuple[str, Any]]) -> dict[Any, Any]:
    return {k: v for (k, v) in x if v is not None}


# --------------------------------------


def load_doc(filename: str) -> BinaryFile:
    with open(filename, "rb") as f:
        doc_json = json.load(f)
    doc: BinaryFile = unserialize(doc_json, BinaryFile)
    return doc


def dump_doc(doc: BinaryFile, filename: str) -> None:
    doc_json = dataclasses.asdict(doc, dict_factory=dict_factory_ignore_none)
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(doc_json, f)


def load_matchresult(filename: str) -> MatchResult:
    with open(filename, "rb") as f:
        matchresult_json = json.load(f)
    matchresult: MatchResult = unserialize(matchresult_json, MatchResult)
    return matchresult


def dump_matchresult(matchresult: MatchResult, filename: str) -> None:
    matchresult_json = dataclasses.asdict(matchresult, dict_factory=dict_factory_ignore_none)
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(matchresult_json, f)


def build_matchresult_with_matchpairs(
    doc1: BinaryFile, doc2: BinaryFile, matched_pair: Iterable[AlgorithmOutputMatchPair]
) -> MatchResult:
    assert doc1.functions is not None
    assert doc2.functions is not None

    matched_addrs1 = set()
    matched_addrs2 = set()
    matches = []

    for m in matched_pair:
        matched_addrs1.add(m.addr_1)
        matched_addrs2.add(m.addr_2)
        matches.append(
            MatchPair(
                function_1=Function(addr=m.addr_1),
                function_2=Function(addr=m.addr_2),
                score=m.score,
            )
        )

    unmatches_1 = [Function(addr=func1.addr) for func1 in doc1.functions if func1.addr not in matched_addrs1]
    unmatches_2 = [Function(addr=func2.addr) for func2 in doc2.functions if func2.addr not in matched_addrs2]

    match_result = MatchResult(
        file_1=BinaryFile(sha256=doc1.sha256, basic_info=BasicInfo(base_address=doc1.basic_info.base_address)),
        file_2=BinaryFile(sha256=doc2.sha256, basic_info=BasicInfo(base_address=doc2.basic_info.base_address)),
        matches=matches,
        unmatches_1=unmatches_1,
        unmatches_2=unmatches_2,
    )

    return match_result


# --------------------------------------


def calculate_code_linecount(code: str) -> int:
    return code.strip().count("\n")


def is_valid_function(func: Function) -> bool:
    if func.linecount is not None:
        return func.linecount >= 7
    else:
        assert func.pseudocode is not None
        return func.pseudocode.strip().count("\n") >= 7


def filter_out_invalid_function(doc1: BinaryFile, doc2: BinaryFile, matchresult: MatchResult) -> MatchResult:
    assert doc1.functions is not None
    assert doc2.functions is not None
    valid_addrs_1 = set(func1.addr for func1 in doc1.functions if is_valid_function(func1))
    valid_addrs_2 = set(func2.addr for func2 in doc2.functions if is_valid_function(func2))

    result = MatchResult(
        file_1=copy.deepcopy(matchresult.file_1),
        file_2=copy.deepcopy(matchresult.file_2),
        matches=[
            copy.deepcopy(p)
            for p in matchresult.matches
            if p.function_1.addr in valid_addrs_1 and p.function_2.addr in valid_addrs_2
        ],
        unmatches_1=[copy.deepcopy(u1) for u1 in matchresult.unmatches_1 if u1.addr in valid_addrs_1],
        unmatches_2=[copy.deepcopy(u2) for u2 in matchresult.unmatches_2 if u2.addr in valid_addrs_2],
    )

    return result
