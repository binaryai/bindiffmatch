from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment as linear_sum_assignment  # type: ignore[import]

from ..models import BinaryFile


class WrapMatrix:
    __slots__ = ["_raw_matrix"]

    def __init__(self, raw_matrix: npt.ArrayLike) -> None:
        self._raw_matrix = np.asarray(raw_matrix)
        assert self._raw_matrix.ndim == 2

    def get_matrix(self) -> npt.NDArray[Any]:
        return self._raw_matrix

    def get_value(self, i: int, j: int) -> Any:
        return self._raw_matrix[i, j]

    def get_raw_matrix(self) -> npt.NDArray[Any]:
        return self._raw_matrix


def wrap_linear_sum_assignment(
    similarity_score_matrix: WrapMatrix,
    maximize: bool = False,
    subrows: Optional[npt.ArrayLike] = None,
    subcols: Optional[npt.ArrayLike] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    if not subrows or not subcols:
        return linear_sum_assignment(similarity_score_matrix.get_raw_matrix(), maximize)  # type: ignore[no-any-return]
    sub_similarity_score_matrix = similarity_score_matrix.get_raw_matrix()[np.ix_(subrows, subcols)]  # type: ignore[arg-type]  # noqa: E501
    row_index_ind, col_index_ind = linear_sum_assignment(sub_similarity_score_matrix, True)
    row_ind = (subrows[i] for i in row_index_ind)  # type: ignore[index]
    col_ind = (subcols[j] for j in col_index_ind)  # type: ignore[index]
    row_ind, col_ind = linear_sum_assignment(similarity_score_matrix.get_raw_matrix(), maximize, subrows, subcols)
    return row_ind, col_ind  # type: ignore[return-value]


def build_similarity_score_matrix(doc1: BinaryFile, doc2: BinaryFile) -> WrapMatrix:
    def get_embeddings_list(doc: BinaryFile) -> list[npt.NDArray[Any]]:
        assert doc.functions is not None
        r = list[npt.NDArray[Any]]()
        for func in doc.functions:
            assert func.embedding is not None
            emb = np.asarray(func.embedding)
            assert emb.ndim == 1
            emb = emb / np.linalg.norm(emb, 2)  # l2 normalize
            emb[emb < -1] = -1
            emb[emb > 1] = 1
            r.append(emb)
        return r

    assert doc1.functions is not None
    assert doc2.functions is not None

    func_count1 = len(doc1.functions)
    func_count2 = len(doc2.functions)
    embedding_len = len(doc1.functions[0].embedding) if func_count1 > 0 else 0  # type: ignore[arg-type]
    embedding_len_2 = len(doc2.functions[0].embedding) if func_count2 > 0 else 0  # type: ignore[arg-type]
    assert embedding_len == embedding_len_2, f"error: embedding lenth diff: {embedding_len} != {embedding_len_2}"

    embeddings_list1 = get_embeddings_list(doc1)
    embeddings1 = np.asarray(embeddings_list1)
    del embeddings_list1
    embeddings_list2 = get_embeddings_list(doc2)
    embeddings2 = np.asarray(embeddings_list2)
    del embeddings_list2

    similarity_score_matrix_raw = embeddings1 @ embeddings2.transpose()
    similarity_score_matrix = WrapMatrix(similarity_score_matrix_raw)
    return similarity_score_matrix
