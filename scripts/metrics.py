import argparse
import dataclasses
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor
import os

from binaryai_bindiffmatch import metricsutils
from binaryai_bindiffmatch.models import Function, MatchResult
from binaryai_bindiffmatch.utils import load_matchresult

global_executor = ProcessPoolExecutor(100)

global_ignore_external_functions = True
global_ignore_small_functions = True


@dataclasses.dataclass(slots=True, kw_only=True)
class MetricValue:
    groundtruth_match_count: int
    groundtruth_unmatch_1_count: int
    groundtruth_unmatch_2_count: int
    groundtruth_ignored_1_count: int
    groundtruth_ignored_2_count: int
    result_match_count: int
    result_unmatch_1_count: int
    result_unmatch_2_count: int
    result_match_in_groundtruth_count: int
    # result_unmatch_1_in_groundtruth_count: int
    # result_unmatch_2_in_groundtruth_count: int
    result_correct_match_count: int

    def get_precision(self) -> float:
        if self.result_match_in_groundtruth_count == 0:
            return float("nan")
        return self.result_correct_match_count / (self.result_match_in_groundtruth_count)

    def get_recall(self) -> float:
        if self.groundtruth_match_count == 0:
            return float("nan")
        return self.result_correct_match_count / self.groundtruth_match_count

    def get_f1(self) -> float:
        precision = self.get_precision()
        recall = self.get_recall()
        if precision + recall == 0:
            return float("nan")
        return 2 * precision * recall / (precision + recall)

    def display(self) -> str:
        groundtruth_file_1_total = (
            self.groundtruth_match_count + self.groundtruth_unmatch_1_count + self.groundtruth_ignored_1_count
        )
        groundtruth_file_2_total = (
            self.groundtruth_match_count + self.groundtruth_unmatch_2_count + self.groundtruth_ignored_2_count
        )
        result_file_1_total = self.result_match_count + self.result_unmatch_1_count
        result_file_2_total = self.result_match_count + self.result_unmatch_2_count
        return f"""\
groundtruth_file_1_total: {groundtruth_file_1_total}, groundtruth_file_2_total: {groundtruth_file_2_total}
result_file_1_total: {result_file_1_total}, result_file_2_total: {result_file_2_total}
groundtruth_match_count = {self.groundtruth_match_count}
result_match_count = {self.result_match_count}
result_match_in_groundtruth_count = {self.result_match_in_groundtruth_count}
result_correct_match_count = {self.result_correct_match_count}
precision = {self.get_precision()}
recall = {self.get_recall()}
f1 = {self.get_f1()}
"""

    def merge(self, another: "MetricValue") -> "MetricValue":
        r = MetricValue(
            groundtruth_match_count=self.groundtruth_match_count + another.groundtruth_match_count,
            groundtruth_unmatch_1_count=self.groundtruth_unmatch_1_count + another.groundtruth_unmatch_1_count,
            groundtruth_unmatch_2_count=self.groundtruth_unmatch_2_count + another.groundtruth_unmatch_2_count,
            groundtruth_ignored_1_count=self.groundtruth_ignored_1_count + another.groundtruth_ignored_1_count,
            groundtruth_ignored_2_count=self.groundtruth_ignored_2_count + another.groundtruth_ignored_2_count,
            result_match_count=self.result_match_count + another.result_match_count,
            result_unmatch_1_count=self.result_unmatch_1_count + another.result_unmatch_1_count,
            result_unmatch_2_count=self.result_unmatch_2_count + another.result_unmatch_2_count,
            result_match_in_groundtruth_count=(
                self.result_match_in_groundtruth_count + another.result_match_in_groundtruth_count
            ),
            result_correct_match_count=self.result_correct_match_count + another.result_correct_match_count,
        )
        return r

    @staticmethod
    def mergelist(values: "Iterable[MetricValue]") -> "MetricValue":
        sum_groundtruth_match_count = 0
        sum_groundtruth_unmatch_1_count = 0
        sum_groundtruth_unmatch_2_count = 0
        sum_groundtruth_ignored_1_count = 0
        sum_groundtruth_ignored_2_count = 0
        sum_result_match_count = 0
        sum_result_unmatch_1_count = 0
        sum_result_unmatch_2_count = 0
        sum_result_match_in_groundtruth_count = 0
        sum_result_correct_match_count = 0
        for v in values:
            sum_groundtruth_match_count += v.groundtruth_match_count
            sum_groundtruth_unmatch_1_count += v.groundtruth_unmatch_1_count
            sum_groundtruth_unmatch_2_count += v.groundtruth_unmatch_2_count
            sum_groundtruth_ignored_1_count += v.groundtruth_ignored_1_count
            sum_groundtruth_ignored_2_count += v.groundtruth_ignored_2_count
            sum_result_match_count += v.result_match_count
            sum_result_unmatch_1_count += v.result_unmatch_1_count
            sum_result_unmatch_2_count += v.result_unmatch_2_count
            sum_result_match_in_groundtruth_count += v.result_match_in_groundtruth_count
            sum_result_correct_match_count += v.result_correct_match_count
        r = MetricValue(
            groundtruth_match_count=sum_groundtruth_match_count,
            groundtruth_unmatch_1_count=sum_groundtruth_unmatch_1_count,
            groundtruth_unmatch_2_count=sum_groundtruth_unmatch_2_count,
            groundtruth_ignored_1_count=sum_groundtruth_ignored_1_count,
            groundtruth_ignored_2_count=sum_groundtruth_ignored_2_count,
            result_match_count=sum_result_match_count,
            result_unmatch_1_count=sum_result_unmatch_1_count,
            result_unmatch_2_count=sum_result_unmatch_2_count,
            result_match_in_groundtruth_count=sum_result_match_in_groundtruth_count,
            result_correct_match_count=sum_result_correct_match_count,
        )
        return r


def evaluation(matchresult: MatchResult, groundtruth: MatchResult) -> MetricValue:
    def normalize_matchresult_addr_1(addr: int) -> int:
        return addr - matchresult.file_1.basic_info.base_address + groundtruth.file_1.basic_info.base_address

    def normalize_matchresult_addr_2(addr: int) -> int:
        return addr - matchresult.file_2.basic_info.base_address + groundtruth.file_2.basic_info.base_address

    def is_external_function(func: Function) -> bool:
        return func.name is not None and func.name.startswith("<EXTERNAL>::")  # XXX

    def is_small_function(func: Function) -> bool:
        return func.linecount is not None and func.linecount <= 6

    def is_valid_function(func: Function) -> bool:
        if global_ignore_external_functions and is_external_function(func):
            return False
        if global_ignore_small_functions and is_small_function(func):
            return False
        return True

    groundtruth_matched_addrpairs = set()
    groundtruth_ignored_1_addrs = set()
    groundtruth_ignored_2_addrs = set()
    groundtruth_unmatch_1_addrs = set()
    groundtruth_unmatch_2_addrs = set()

    for p in groundtruth.matches:
        addr1 = p.function_1.addr
        addr2 = p.function_2.addr
        valid_1 = is_valid_function(p.function_1)
        valid_2 = is_valid_function(p.function_2)
        if valid_1 and valid_2:  # only keep match with BOTH funcs are valid
            groundtruth_matched_addrpairs.add((addr1, addr2))
        else:  # ignore invalid functions
            groundtruth_ignored_1_addrs.add(addr1)
            groundtruth_ignored_2_addrs.add(addr2)
    for f in groundtruth.unmatches_1:
        addr = f.addr
        valid = is_valid_function(f)
        if valid:
            groundtruth_unmatch_1_addrs.add(addr)
        else:
            groundtruth_ignored_1_addrs.add(addr)
    for f in groundtruth.unmatches_2:
        addr = f.addr
        valid = is_valid_function(f)
        if valid:
            groundtruth_unmatch_2_addrs.add(addr)
        else:
            groundtruth_ignored_2_addrs.add(addr)

    groundtruth_file_1_valid_addrs = set(a for a, _ in groundtruth_matched_addrpairs) | groundtruth_unmatch_1_addrs
    groundtruth_file_2_valid_addrs = set(b for _, b in groundtruth_matched_addrpairs) | groundtruth_unmatch_2_addrs

    groundtruth_match_count = len(groundtruth_matched_addrpairs)
    groundtruth_unmatch_1_count = len(groundtruth_unmatch_1_addrs)
    groundtruth_unmatch_2_count = len(groundtruth_unmatch_2_addrs)
    groundtruth_ignored_1_count = len(groundtruth_ignored_1_addrs)
    groundtruth_ignored_2_count = len(groundtruth_ignored_2_addrs)

    result_match_count = len(matchresult.matches)
    result_unmatch_1_count = len(matchresult.unmatches_1)
    result_unmatch_2_count = len(matchresult.unmatches_2)
    result_match_in_groundtruth_count = 0
    result_correct_match_count = 0

    result_match_different_addr1_count = len(set(p.function_1.addr for p in matchresult.matches))
    result_match_different_addr2_count = len(set(p.function_2.addr for p in matchresult.matches))
    if not (result_match_count == result_match_different_addr1_count == result_match_different_addr2_count):
        # print(
        #     "[warning]: count mismatch",
        #     result_match_count,
        #     result_match_different_addr1_count,
        #     result_match_different_addr2_count,
        # )
        pass

    for p in matchresult.matches:
        addr1 = normalize_matchresult_addr_1(p.function_1.addr)
        addr2 = normalize_matchresult_addr_2(p.function_2.addr)
        if addr1 not in groundtruth_file_1_valid_addrs or addr2 not in groundtruth_file_2_valid_addrs:
            # only matches with two addrs BOTH contains in groundtruth will be considered
            continue
        result_match_in_groundtruth_count += 1
        if (addr1, addr2) in groundtruth_matched_addrpairs:
            result_correct_match_count += 1

    r = MetricValue(
        groundtruth_match_count=groundtruth_match_count,
        groundtruth_unmatch_1_count=groundtruth_unmatch_1_count,
        groundtruth_unmatch_2_count=groundtruth_unmatch_2_count,
        groundtruth_ignored_1_count=groundtruth_ignored_1_count,
        groundtruth_ignored_2_count=groundtruth_ignored_2_count,
        result_match_count=result_match_count,
        result_unmatch_1_count=result_unmatch_1_count,
        result_unmatch_2_count=result_unmatch_2_count,
        result_match_in_groundtruth_count=result_match_in_groundtruth_count,
        result_correct_match_count=result_correct_match_count,
    )
    return r


def evaluation_on_matchresultfile(matchresult_file: str, groundtruth_file: str) -> MetricValue:
    matchresult = load_matchresult(matchresult_file)
    groundtruth = load_matchresult(groundtruth_file)
    result = evaluation(matchresult, groundtruth)
    return result


def batch_evaluation_files(filepairs: Iterable[tuple[str, str]]) -> list[MetricValue]:
    tasks: list[Future[MetricValue]] = []
    for algoresult_file, groundtruth_file in filepairs:
        task = global_executor.submit(evaluation_on_matchresultfile, algoresult_file, groundtruth_file)
        tasks.append(task)
    results = [task.result() for task in tasks]
    return results


def evaluation_on_testcase(datadir: str, algorithm: str) -> None:
    library_filenames: dict[str, list[str]] = {}
    filepairs: list[tuple[str, str]] = []
    for (library1, version1, optimazation1), (library2, version2, optimazation2) in metricsutils.testcase_pairs:
        assert library1 == library2
        library = library1
        if library in library_filenames:
            filenames = library_filenames[library]
        else:
            filenames = metricsutils.get_stripped_filenames(datadir, library)
            library_filenames[library] = filenames
        for filename in filenames:
            testcase_matchresult_filename = metricsutils.get_algorithm_matchresult_relpath(
                algorithm, library1, version1, optimazation1, library2, version2, optimazation2, filename
            )
            groundtruth_matchresult_filename = metricsutils.get_algorithm_matchresult_relpath(
                "groundtruth", library1, version1, optimazation1, library2, version2, optimazation2, filename
            )
            matchresult_file = os.path.join(datadir, testcase_matchresult_filename)
            groundtruth_file = os.path.join(datadir, groundtruth_matchresult_filename)
            if os.path.exists(matchresult_file) and os.path.exists(groundtruth_file):    
                filepairs.append(
                    (
                        os.path.join(datadir, testcase_matchresult_filename),
                        os.path.join(datadir, groundtruth_matchresult_filename),
                    )
                )
            else:
                # print(f"warning: file `{matchresult_file}` or `{groundtruth_file}` not found")
                pass
    results = batch_evaluation_files(filepairs)
    metricresult = MetricValue.mergelist(results)
    print(metricresult.display())


def evaluation_on_example(datadir: str, algorithm: str) -> None:
    (library1, version1, optimazation1), (library2, version2, optimazation2) = metricsutils.example_pair
    filename = metricsutils.example_filename
    example_matchresult_filename = metricsutils.get_algorithm_matchresult_relpath(
        algorithm, library1, version1, optimazation1, library2, version2, optimazation2, filename
    )
    groundtruth_matchresult_filename = metricsutils.get_algorithm_matchresult_relpath(
        "groundtruth", library1, version1, optimazation1, library2, version2, optimazation2, filename
    )
    metricresult = evaluation_on_matchresultfile(
        os.path.join(datadir, example_matchresult_filename), os.path.join(datadir, groundtruth_matchresult_filename)
    )
    print(metricresult.display())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("choose", choices=["testcases", "example"])
    parser.add_argument("algorithm", choices=["diaphora", "binaryai"])
    parser.add_argument("--datadir", required=False, default="./data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    choose = args.choose
    algorithm = args.algorithm
    datadir = args.datadir

    match choose:
        case "testcases":
            evaluation_on_testcase(datadir, algorithm)
        case "example":
            evaluation_on_example(datadir, algorithm)
        case _:
            print(f"[error] unknown choose `{choose}`")


if __name__ == "__main__":
    main()
