import math
import platform
import tempfile
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from nanolsap import linear_sum_assignment

from ..models import BinaryFile

USE_MEMMAP_NUMPY_ARRAY = False if platform.system() == "Windows" else True
WRAP_MATRIX_SCALE: int = 127
WRAP_MATRIX_DTYPE = np.int8
BLOCK_MATMUL_UNIT_LIMIT = 5000


class WrapMatrix:
    __slots__ = ["_raw_matrix", "_transpose", "_scale"]

    def __init__(self, raw_matrix: npt.ArrayLike, *, transpose: bool = False, scale: int = 1) -> None:
        self._raw_matrix = np.asarray(raw_matrix)
        self._transpose = transpose
        self._scale = scale
        assert self._raw_matrix.ndim == 2
        assert self._scale != 0

    def get_matrix(self) -> npt.NDArray[Any]:
        if self._transpose:
            r = self._raw_matrix.transpose()
        else:
            r = self._raw_matrix
        if self._scale != 1:
            r = r / self._scale
        return r

    def get_value(self, i: int, j: int) -> Any:
        if self._transpose:
            return self._raw_matrix[j, i] / self._scale
        return self._raw_matrix[i, j] / self._scale

    def get_raw_matrix(self) -> npt.NDArray[Any]:
        return self._raw_matrix

    def is_transpose(self) -> bool:
        return self._transpose


def wrap_linear_sum_assignment(
    similarity_score_matrix: WrapMatrix,
    maximize: bool = False,
    subrows: Optional[npt.ArrayLike] = None,
    subcols: Optional[npt.ArrayLike] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # nanolsap.linear_sum_assignment works much slower when nr > nc, so we manually do a shadow transpose
    if similarity_score_matrix.is_transpose():
        col_ind, row_ind = linear_sum_assignment(similarity_score_matrix.get_raw_matrix(), maximize, subcols, subrows)
    else:
        row_ind, col_ind = linear_sum_assignment(similarity_score_matrix.get_raw_matrix(), maximize, subrows, subcols)
    return row_ind, col_ind


def create_numpy_array(
    data: npt.ArrayLike | None, dtype: npt.DTypeLike, shape: int | tuple[int, ...]
) -> npt.NDArray[Any]:
    def prod(d: int | tuple[int, ...]) -> int:
        if isinstance(d, int):
            return d
        r = 1
        for c in d:
            r *= c
        return r

    if not USE_MEMMAP_NUMPY_ARRAY:
        a = np.empty(shape=shape, dtype=dtype)
    else:
        with tempfile.NamedTemporaryFile(prefix="binaryai_bindiffmatch_tmp_") as fp:
            memsize = np.dtype(dtype).itemsize * prod(shape)
            fp.truncate(memsize)
            a = np.memmap(fp.name, dtype=dtype, shape=shape)

    if data is not None:
        a[:] = data
    return a


def block_matmul(
    x1: npt.ArrayLike,
    x2: npt.ArrayLike,
    *,
    out: npt.NDArray[Any] | None = None,
    out_dtype: npt.DTypeLike = np.float64,
    unit_limit: int = 0,
) -> npt.NDArray[Any]:
    a = np.asarray(x1)
    b = np.asarray(x2)
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    n, m = a.shape[0], b.shape[1]
    u = max(n, m) if unit_limit == 0 else unit_limit
    r = np.empty(shape=(n, m), dtype=out_dtype) if out is None else out
    for i in range((n + u - 1) // u):
        for j in range((m + u - 1) // u):
            tmp_x = a[i * u : (i + 1) * u, :]
            tmp_y = b[:, j * u : (j + 1) * u]
            r[i * u : (i + 1) * u, j * u : (j + 1) * u] = tmp_x @ tmp_y
    return r


def build_similarity_score_matrix(doc1: BinaryFile, doc2: BinaryFile) -> WrapMatrix:
    def get_embeddings_list_with_scale(doc: BinaryFile, scale: int = 1) -> list[npt.NDArray[Any]]:
        assert doc.functions is not None
        r = list[npt.NDArray[Any]]()
        for func in doc.functions:
            assert func.embedding is not None
            emb = np.asarray(func.embedding)
            assert emb.ndim == 1
            emb = emb / np.linalg.norm(emb, 2)  # l2 normalize
            emb[emb < -1] = -1
            emb[emb > 1] = 1
            if scale != 1:
                sqrt_scale = math.sqrt(scale)
                # notice this sqrt, because later we will multiply the two embedding matrixes
                emb = emb * sqrt_scale
            r.append(emb)
        return r

    assert doc1.functions is not None
    assert doc2.functions is not None

    func_count1 = len(doc1.functions)
    func_count2 = len(doc2.functions)
    embedding_len = len(doc1.functions[0].embedding) if func_count1 > 0 else 0  # type: ignore[arg-type]
    embedding_len_2 = len(doc2.functions[0].embedding) if func_count2 > 0 else 0  # type: ignore[arg-type]
    assert embedding_len == embedding_len_2, f"error: embedding lenth diff: {embedding_len} != {embedding_len_2}"

    transpose = func_count1 > func_count2
    scale = WRAP_MATRIX_SCALE
    tmp_dtype = np.float32
    wrap_dtype = WRAP_MATRIX_DTYPE
    tmp_wrap_dtype = wrap_dtype

    scale_embeddings_list1 = get_embeddings_list_with_scale(doc1, scale)
    embeddings1 = create_numpy_array(scale_embeddings_list1, tmp_dtype, (func_count1, embedding_len))
    del scale_embeddings_list1
    scale_embeddings_list2 = get_embeddings_list_with_scale(doc2, scale)
    embeddings2 = create_numpy_array(scale_embeddings_list2, tmp_dtype, (func_count2, embedding_len))
    del scale_embeddings_list2

    if transpose:
        tmp_func_count1, tmp_func_count2 = func_count2, func_count1
        tmp_embeddings1, tmp_embeddings2 = embeddings2, embeddings1
    else:
        tmp_func_count1, tmp_func_count2 = func_count1, func_count2
        tmp_embeddings1, tmp_embeddings2 = embeddings1, embeddings2
    similarity_score_matrix_raw = create_numpy_array(None, tmp_wrap_dtype, (tmp_func_count1, tmp_func_count2))
    # notice: here, the matmul may cause memory usage sudden increase then decrease
    # Only when two input matrixes and the output matrixes has same dtype, numpy can do in place calculate.
    # otherwise, if output dtype is not same as input dtype, numpy will do internal memory copy on input matrixes,
    # so it cannot benefit from memmap memory swap to saving memory, and may cause oom on huge input matrixes.
    # np.matmul(tmp_embeddings1, tmp_embeddings2.transpose(), out=similarity_score_matrix_raw, casting="unsafe")
    block_matmul(
        tmp_embeddings1,
        tmp_embeddings2.transpose(),
        out=similarity_score_matrix_raw,
        out_dtype=tmp_wrap_dtype,
        unit_limit=BLOCK_MATMUL_UNIT_LIMIT,
    )

    del tmp_embeddings1
    del tmp_embeddings2
    del embeddings1
    del embeddings2

    similarity_score_matrix_raw = similarity_score_matrix_raw.astype(wrap_dtype, copy=False)
    similarity_score_matrix = WrapMatrix(similarity_score_matrix_raw, transpose=transpose, scale=scale)
    return similarity_score_matrix
