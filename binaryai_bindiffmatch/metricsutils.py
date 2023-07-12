# import functools
# import itertools
import os
from collections.abc import Generator

"""
stripped raw binaries: (filename ends with .strip)
{data}/files/{library}/stripped_binaries/{library}-{version}-{optimazation}/{filename}
doc with embedding and funcname label: (functions powered by Ghidra, embeddings powered by BinaryAI BAI-2.0 model)
{data}/labeleds/{library}/stripped_binaries/{library}-{version}-{optimazation}/{filename}.json
diaphora exported database:
{data}/diaphora_sqlites/{library}/striiped_binaries/{library}-{version}-{optimazation}/{filename}.sqlite

groundtruth matchresult:
{data}/matchresults/groundtruth/{library}-{version}-{optimazation}__vs__{library}-{version}-{optimazation}/{filename}.json
diaphora matchresult:
{data}/matchresults/diaphora/{library}-{version}-{optimazation}__vs__{library}-{version}-{optimazation}/{filename}.json
binaryai matchresult:
{data}/matchresults/binaryai/{library}-{version}-{optimazation}__vs__{library}-{version}-{optimazation}/{filename}.json
"""

libraries = ["coreutils", "diffutils", "findutils"]
versions = {
    "coreutils": ["5.93", "6.4", "7.6", "8.1", "8.30"],
    "diffutils": ["2.8", "3.1", "3.4", "3.6"],
    "findutils": ["4.233", "4.41", "4.6"],
    "openssl": ["1.1.1u", "3.1.1"],
}
optimazations = {
    "coreutils": ["O0", "O1", "O2", "O3"],
    "diffutils": ["O0", "O1", "O2", "O3"],
    "findutils": ["O0", "O1", "O2", "O3"],
    "openssl": ["gcc_x64_O3", "gcc_arm_O0"],
}


def build_testcase_cross_version_pairs_on_library(
    library: str,
) -> Generator[tuple[tuple[str, str, str], tuple[str, str, str]], None, None]:
    optimazation = "O1"
    to_compare_version = versions[library][-1]
    for version in versions[library][:-1]:
        assert version != to_compare_version, (version, to_compare_version)
        yield ((library, version, optimazation), (library, to_compare_version, optimazation))


def build_testcase_cross_optimization_pairs_on_library(
    library: str,
) -> Generator[tuple[tuple[str, str, str], tuple[str, str, str]], None, None]:
    for version in versions[library]:
        to_compare_optimazation = "O3"
        for optimazation in optimazations[library][:-1]:
            assert optimazation != to_compare_optimazation, (optimazation, to_compare_optimazation)
            yield ((library, version, optimazation), (library, version, to_compare_optimazation))


def get_stripped_filenames(datadir: str, library: str) -> list[str]:
    # assert library in libraries
    stripped_binary_relpath = get_stripped_binary_relpath(
        library, versions[library][-1], optimazations[library][-1], None
    )
    result = []
    for filename in os.listdir(os.path.join(datadir, stripped_binary_relpath)):
        assert filename.endswith(".strip")
        result.append(filename)
    return result


def get_stripped_binary_relpath(library: str, version: str, optimazation: str, filename: str | None) -> str:
    return f"files/{library}/stripped_binaries/{library}-{version}-{optimazation}/{filename if filename else ''}"


def get_labeled_doc_relpath(library: str, version: str, optimazation: str, filename: str | None) -> str:
    return f"labeleds/{library}/stripped_binaries/{library}-{version}-{optimazation}/{filename+'.json' if filename else ''}"  # noqa: E501


def get_groundtruth_matchresult_relpath(
    library: str, version1: str, optimazation1: str, version2: str, optimazation2: str, filename: str | None
) -> str:
    return f"matchresults/groundtruth/{library}/{library}-{version1}-{optimazation1}__vs__{library}-{version2}-{optimazation2}/{filename+'.json' if filename else ''}"  # noqa: E501


def get_diaphora_matchresult_relpath(
    library: str, version1: str, optimazation1: str, version2: str, optimazation2: str, filename: str | None
) -> str:  # noqa: E501
    return f"matchresults/diaphora/{library}/{library}-{version1}-{optimazation1}__vs__{library}-{version2}-{optimazation2}/{filename+'.json' if filename else ''}"  # noqa: E501


def get_algorithm_matchresult_relpath(
    algorithm: str,
    library1: str,
    version1: str,
    optimazation1: str,
    library2: str,
    version2: str,
    optimazation2: str,
    filename: str | None,
) -> str:
    library = library1 if library1 == library2 else "_other"
    return f"matchresults/{algorithm}/{library}/{library1}-{version1}-{optimazation1}__vs__{library2}-{version2}-{optimazation2}/{filename+'.json' if filename else ''}"  # noqa: E501


def get_matchresult_cross_vs_named_filepair(
    library: str,
    version1: str,
    optimazation1: str,
    version2: str,
    optimazation2: str,
) -> str:  # noqa: E501
    return f"{library}-{version1}-{optimazation1}__vs__{library}-{version2}-{optimazation2}"


coreutils_cross_version_pairs = list(build_testcase_cross_version_pairs_on_library("coreutils"))
findutils_cross_version_pairs = list(build_testcase_cross_version_pairs_on_library("findutils"))
diffutils_cross_version_pairs = list(build_testcase_cross_version_pairs_on_library("diffutils"))

coreutils_cross_optimization_pairs = list(build_testcase_cross_optimization_pairs_on_library("coreutils"))
findutils_cross_optimization_pairs = list(build_testcase_cross_optimization_pairs_on_library("findutils"))
diffutils_cross_optimization_pairs = list(build_testcase_cross_optimization_pairs_on_library("diffutils"))


testcase_pairs = (
    coreutils_cross_version_pairs
    + findutils_cross_version_pairs
    + diffutils_cross_version_pairs
    + coreutils_cross_optimization_pairs
    + findutils_cross_optimization_pairs
    + diffutils_cross_optimization_pairs
)
example_pair = (("openssl", "1.1.1u", "gcc_arm_O0"), ("openssl", "3.1.1", "gcc_x64_O3"))
example_filename = "openssl.strip"
