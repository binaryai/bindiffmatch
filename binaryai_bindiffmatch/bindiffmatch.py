import itertools
from collections.abc import Iterable
from typing import Any

import networkx as nx  # type: ignore[import]
import numpy as np

from .models import AlgorithmOutputMatchPair, BinaryFile, MatchResult
from .utils import build_matchresult_with_matchpairs, is_valid_function

try:
    from .similarity_matrix.lowmem import (
        WrapMatrix,
        build_similarity_score_matrix,
        create_numpy_array,
        wrap_linear_sum_assignment,
    )
except ImportError:
    from .similarity_matrix.basic import (  # type: ignore
        WrapMatrix,
        build_similarity_score_matrix,
        create_numpy_array,
        wrap_linear_sum_assignment,
    )


def build_callgraph(doc: BinaryFile, *, with_attributes: bool = True) -> nx.DiGraph:
    assert doc.functions is not None
    g = nx.DiGraph()
    func_count = len(doc.functions)
    g.graph["func_count"] = func_count
    for i, func in enumerate(doc.functions):
        assert func.embedding is not None
        if with_attributes:
            g.add_node(func.addr, **{"id": i})
        else:
            g.add_node(func.addr)
    for func in doc.functions:
        addr = func.addr
        if func.callees:
            for callee in func.callees:
                assert g.has_node(addr), addr
                assert g.has_node(callee), callee
                g.add_edge(addr, callee)
    assert g.number_of_nodes() == func_count
    return g


def filter_mwm_match_results(
    doc1: BinaryFile,
    doc2: BinaryFile,
    similarity_score_matrix: WrapMatrix,
    row_ind: Iterable[int],
    col_ind: Iterable[int],
    *,
    threshold: float,
) -> list[AlgorithmOutputMatchPair]:
    assert doc1.functions is not None
    assert doc2.functions is not None
    matched = []
    for id1, id2 in zip(row_ind, col_ind):
        score = float(similarity_score_matrix.get_value(id1, id2))
        if score >= threshold:
            func1 = doc1.functions[id1]
            func2 = doc2.functions[id2]
            addr1 = func1.addr
            addr2 = func2.addr
            if is_valid_function(func1) and is_valid_function(func2):  # XXX
                matched.append(
                    AlgorithmOutputMatchPair(
                        addr_1=addr1,
                        addr_2=addr2,
                        score=round(score, 2),
                    )
                )
    return matched


def do_mwm_on_full_matrix(
    doc1: BinaryFile, doc2: BinaryFile, similarity_score_matrix: WrapMatrix, *, threshold: float
) -> list[AlgorithmOutputMatchPair]:
    assert doc1.functions is not None
    assert doc2.functions is not None
    row_ind, col_ind = wrap_linear_sum_assignment(similarity_score_matrix, True)
    matched = filter_mwm_match_results(doc1, doc2, similarity_score_matrix, row_ind, col_ind, threshold=threshold)
    return matched


def do_mwm_on_sub_rows_cols(
    doc1: BinaryFile,
    doc2: BinaryFile,
    similarity_score_matrix: WrapMatrix,
    subrowindexes: Iterable[int],
    subcolindexes: Iterable[int],
    *,
    threshold: float,
) -> list[AlgorithmOutputMatchPair]:
    assert doc1.functions is not None
    assert doc2.functions is not None

    subrowindexes = subrowindexes if isinstance(subrowindexes, list) else list(subrowindexes)
    subcolindexes = subcolindexes if isinstance(subcolindexes, list) else list(subcolindexes)
    if not subrowindexes or not subcolindexes:
        return []
    row_ind, col_ind = wrap_linear_sum_assignment(similarity_score_matrix, True, subrowindexes, subcolindexes)

    matched = filter_mwm_match_results(doc1, doc2, similarity_score_matrix, row_ind, col_ind, threshold=threshold)
    return matched


def do_mwm_on_sub_pairs(
    doc1: BinaryFile,
    doc2: BinaryFile,
    similarity_score_matrix: WrapMatrix,
    subnodeindexpairs: Iterable[tuple[int, int]],  # if duplicated, only process once
    *,
    threshold: float,
) -> list[AlgorithmOutputMatchPair]:
    assert doc1.functions is not None
    assert doc2.functions is not None

    subnodeindexpairs = subnodeindexpairs if isinstance(subnodeindexpairs, list) else list(subnodeindexpairs)

    subrowindexes = list(set(id1 for id1, _ in subnodeindexpairs))
    subcolindexes = list(set(id2 for _, id2 in subnodeindexpairs))
    subrowindex_map = dict(zip(subrowindexes, itertools.count()))
    subcolindex_map = dict(zip(subcolindexes, itertools.count()))

    sub_similarity_score_matrix = create_numpy_array(-1, np.float32, (len(subrowindexes), len(subcolindexes)))
    for id1, id2 in subnodeindexpairs:
        id1_index, id2_index = subrowindex_map[id1], subcolindex_map[id2]
        sub_similarity_score_matrix[id1_index, id2_index] = similarity_score_matrix.get_value(id1, id2)

    row_index_ind, col_index_ind = wrap_linear_sum_assignment(WrapMatrix(sub_similarity_score_matrix), True)
    row_ind = (subrowindexes[i] for i in row_index_ind)
    col_ind = (subcolindexes[j] for j in col_index_ind)

    matched = filter_mwm_match_results(doc1, doc2, similarity_score_matrix, row_ind, col_ind, threshold=threshold)
    return matched


def get_callrelation_neighbors(g: nx.DiGraph, root_node: Any, *, hop: int) -> tuple[Iterable[Any], Iterable[Any]]:
    def get_callrelation_neighbors_internal(
        g: nx.DiGraph, root_node: Any, *, hop: int, reverse: bool = False
    ) -> Iterable[Any]:
        nodes_set = set()
        seen_nodes = {root_node}
        layer_nodes = {root_node}
        for _ in range(hop):
            new_layer_nodes = set()
            for layer_node in layer_nodes:
                for e in g.in_edges(layer_node) if reverse else g.out_edges(layer_node):
                    next_layer_node = e[0] if reverse else e[1]
                    if next_layer_node not in seen_nodes:
                        seen_nodes.add(next_layer_node)
                        new_layer_nodes.add(next_layer_node)
            nodes_set.update(new_layer_nodes)
            layer_nodes = new_layer_nodes
        return nodes_set

    outedge_nodes_set = get_callrelation_neighbors_internal(g, root_node, hop=hop, reverse=False)
    inedge_nodes_set = get_callrelation_neighbors_internal(g, root_node, hop=hop, reverse=True)

    return outedge_nodes_set, inedge_nodes_set


def do_spread(
    doc1: BinaryFile,
    doc2: BinaryFile,
    similarity_score_matrix: WrapMatrix,
    initial_matchpairs: list[AlgorithmOutputMatchPair],
    *,
    hop: int,
    threshold: float,
) -> list[AlgorithmOutputMatchPair]:
    assert doc1.functions is not None
    assert doc2.functions is not None

    g1 = build_callgraph(doc1, with_attributes=False)
    g2 = build_callgraph(doc2, with_attributes=False)
    addr_to_id_map1 = {f.addr: i for i, f in enumerate(doc1.functions)}
    addr_to_id_map2 = {f.addr: i for i, f in enumerate(doc2.functions)}

    processed_addrs_1 = set()
    processed_addrs_2 = set()
    for matchpair in initial_matchpairs:
        processed_addrs_1.add(matchpair.addr_1)
        processed_addrs_2.add(matchpair.addr_2)

    spread_matchpairs = list[AlgorithmOutputMatchPair]()

    turn = 0
    while True:
        turn += 1

        subnodepairs = []
        for matchpair in initial_matchpairs:
            neighbors_1_out, neighbors_1_in = get_callrelation_neighbors(g1, matchpair.addr_1, hop=hop)
            neighbors_2_out, neighbors_2_in = get_callrelation_neighbors(g2, matchpair.addr_2, hop=hop)
            new_neighbors_1_out = [addr1 for addr1 in neighbors_1_out if addr1 not in processed_addrs_1]
            new_neighbors_1_in = [addr1 for addr1 in neighbors_1_in if addr1 not in processed_addrs_1]
            new_neighbors_2_out = [addr2 for addr2 in neighbors_2_out if addr2 not in processed_addrs_2]
            new_neighbors_2_in = [addr2 for addr2 in neighbors_2_in if addr2 not in processed_addrs_2]
            for addr1, addr2 in itertools.chain(
                itertools.product(new_neighbors_1_out, new_neighbors_2_out),
                itertools.product(new_neighbors_1_in, new_neighbors_2_in),
            ):
                id1 = addr_to_id_map1[addr1]
                id2 = addr_to_id_map2[addr2]
                subnodepairs.append((id1, id2))

                processed_addrs_1.add(addr1)
                processed_addrs_2.add(addr2)

        if not subnodepairs:
            break

        new_matchpairs = do_mwm_on_sub_pairs(doc1, doc2, similarity_score_matrix, subnodepairs, threshold=threshold)

        spread_matchpairs.extend(new_matchpairs)
        initial_matchpairs = new_matchpairs

    return spread_matchpairs


def get_remained_nodepair_indexes(
    doc1: BinaryFile, doc2: BinaryFile, matched_pairs: Iterable[AlgorithmOutputMatchPair]
) -> tuple[list[int], list[int]]:
    assert doc1.functions is not None
    assert doc2.functions is not None
    addr1_inpair = set()
    addr2_inpair = set()
    for m in matched_pairs:
        addr1_inpair.add(m.addr_1)
        addr2_inpair.add(m.addr_2)
    index1_notinpair = [i for i, f in enumerate(doc1.functions) if f.addr not in addr1_inpair]
    index2_notinpair = [i for i, f in enumerate(doc2.functions) if f.addr not in addr2_inpair]
    return index1_notinpair, index2_notinpair


def do_bindiffmatch(
    doc1: BinaryFile,
    doc2: BinaryFile,
    *,
    threshold_high: float,
    threshold_low: float,
    threshold_remain: float,
    hop: int,
) -> MatchResult:
    # prepare
    similarity_score_matrix = build_similarity_score_matrix(doc1, doc2)

    # stage 1
    initial_matchpairs = do_mwm_on_full_matrix(doc1, doc2, similarity_score_matrix, threshold=threshold_high)

    # stage 2
    spread_matchpairs = do_spread(
        doc1, doc2, similarity_score_matrix, initial_matchpairs, hop=hop, threshold=threshold_low
    )

    # stage 3
    remained_nodes1, remained_nodes2 = get_remained_nodepair_indexes(
        doc1, doc2, itertools.chain(initial_matchpairs, spread_matchpairs)
    )
    remained_matchpairs = do_mwm_on_sub_rows_cols(
        doc1, doc2, similarity_score_matrix, remained_nodes1, remained_nodes2, threshold=threshold_remain
    )

    # merge
    matchresult = build_matchresult_with_matchpairs(
        doc1, doc2, itertools.chain(initial_matchpairs, spread_matchpairs, remained_matchpairs)
    )
    return matchresult
