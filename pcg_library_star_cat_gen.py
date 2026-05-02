"""pcg_library_star_cat_gen.py

Importable exact PCG checker for graph6 inputs.

Pipeline
--------
For each input graph, try the following exact witness families in order:
    1) star-PCG,
    2) caterpillar-PCG,
    3) general PCG via a tree-metric distance matrix + four-point condition.

All solver variables are unbounded integers; no artificial maximum distance bound
is imposed.

Accepted inputs
---------------
- a single graph6 string,
- a list / tuple of graph6 strings,
- a filename / Path containing one graph6 string per line.

Primary entry points
--------------------
- analyze(source, verbose=False) -> list[dict]
- analyze_one_graph6(g6, verbose=False) -> dict
- analyze_graph6_list(graph6_list, verbose=False) -> list[dict]
- analyze_graph6_file(path, verbose=False) -> list[dict]

Result schema
-------------
{
    "n": 7,
    "graph6": "F?Beo",
    "status": "sat-star" | "sat-caterpillar" | "sat-general" | "unsat",
    "mode": "exact",
    "edges": [[...], ...],
    "interval": [L, U] | None,
    "distance_matrix": [[...], ...] | None,
    "time_elapsed": float,
    "note": None | str,
}
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import networkx as nx
from z3 import And, If, Int, IntVal, Or, Then, sat, unsat

__all__ = [
    "analyze",
    "analyze_one_graph6",
    "analyze_graph6_list",
    "analyze_graph6_file",
    "results_to_jsonable",
]

JsonDict = dict
Graph6Source = Union[str, Path, Sequence[str]]
PairList = List[Tuple[int, int, bool]]


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[pcg_library] {message}")


def _new_solver():
    """Create a lightly-preprocessed exact SMT solver."""
    return Then("simplify", "solve-eqs", "smt").solver()


def _abs_z3(expr):
    return If(expr >= 0, expr, -expr)


def _graph_from_graph6(g6: str) -> nx.Graph:
    g6 = g6.strip()
    if not g6:
        raise ValueError("Empty graph6 string.")
    try:
        G = nx.from_graph6_bytes(g6.encode())
    except Exception as e:
        raise ValueError(f"Invalid graph6 string: {g6!r}. Details: {e}") from e
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def _edges_of_graph(G: nx.Graph) -> List[List[int]]:
    return [[int(u), int(v)] for u, v in sorted(G.edges())]


def _read_graph6_file(path: Union[str, Path], verbose: bool = False) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    _log(verbose, f"Reading graph6 strings from file: {file_path}")
    out: List[str] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(">"):
                continue
            out.append(line)
    _log(verbose, f"Loaded {len(out)} graph6 string(s) from file.")
    return out


def _normalize_source(source: Graph6Source, verbose: bool = False) -> List[str]:
    if isinstance(source, (list, tuple)):
        out = [str(x).strip() for x in source if str(x).strip()]
        _log(verbose, f"Source interpreted as an in-memory list with {len(out)} graph(s).")
        return out

    if isinstance(source, Path):
        return _read_graph6_file(source, verbose=verbose)

    if isinstance(source, str):
        candidate = Path(source)
        if candidate.exists() and candidate.is_file():
            return _read_graph6_file(candidate, verbose=verbose)
        g6 = source.strip()
        if not g6:
            return []
        _log(verbose, "Source interpreted as a single graph6 string.")
        return [g6]

    raise TypeError(
        "source must be a graph6 string, a list/tuple of graph6 strings, or a filename/path"
    )


def _pair_data(G: nx.Graph) -> PairList:
    n = G.number_of_nodes()
    has_edge = G.has_edge
    return [(i, j, has_edge(i, j)) for i in range(n) for j in range(i + 1, n)]


def _check_interval_against_graph(
    pairs: PairList,
    distance_matrix: List[List[int]],
    interval: List[int],
) -> bool:
    L, U = interval
    for i, j, is_edge in pairs:
        in_interval = L <= distance_matrix[i][j] <= U
        if is_edge != in_interval:
            return False
    return True


def _build_star_distance_matrix(p_values: List[int]) -> List[List[int]]:
    n = len(p_values)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = p_values[i] + p_values[j]
            dist[i][j] = dij
            dist[j][i] = dij
    return dist


def _build_caterpillar_distance_matrix(xs: List[int], ps: List[int]) -> List[List[int]]:
    n = len(xs)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = ps[i] + ps[j] + abs(xs[i] - xs[j])
            dist[i][j] = dij
            dist[j][i] = dij
    return dist


def _solve_star_pcg_exact(G: nx.Graph, pairs: PairList, verbose: bool = False) -> JsonDict:
    n = G.number_of_nodes()
    _log(verbose, f"Starting star-PCG exact solve for n={n}, m={G.number_of_edges()}.")

    solver = _new_solver()
    p = [Int(f"sp_{i}") for i in range(n)]
    L = Int("sL")
    U = Int("sU")

    for i in range(n):
        solver.add(p[i] >= 1)
    solver.add(L >= 0)
    solver.add(U >= L)

    for i, j, is_edge in pairs:
        dij = p[i] + p[j]
        if is_edge:
            solver.add(dij >= L, dij <= U)
        else:
            solver.add(Or(dij < L, dij > U))

    _log(verbose, "Star constraints added. Invoking Z3.")
    status = solver.check()
    _log(verbose, f"Star solver status: {status}")

    if status == sat:
        model = solver.model()
        interval = [
            model.evaluate(L, model_completion=True).as_long(),
            model.evaluate(U, model_completion=True).as_long(),
        ]
        p_values = [model.evaluate(p[i], model_completion=True).as_long() for i in range(n)]
        distance_matrix = _build_star_distance_matrix(p_values)
        if not _check_interval_against_graph(pairs, distance_matrix, interval):
            raise RuntimeError("Internal error: star witness does not validate against the graph.")
        _log(verbose, f"Star witness found with interval={interval}.")
        _log(verbose, f"Star pendant lengths p={p_values}")
        return {
            "status": "sat-star",
            "interval": interval,
            "distance_matrix": distance_matrix,
            "note": None,
        }

    if status == unsat:
        _log(verbose, "Graph is not star-PCG under the exact encoding.")
        return {
            "status": "unsat",
            "interval": None,
            "distance_matrix": None,
            "note": None,
        }

    reason = solver.reason_unknown()
    _log(verbose, f"Star solver returned unknown: {reason}")
    raise RuntimeError(f"star solver returned unknown: {reason}")


def _solve_caterpillar_pcg_exact(G: nx.Graph, pairs: PairList, verbose: bool = False) -> JsonDict:
    n = G.number_of_nodes()
    _log(verbose, f"Starting caterpillar-PCG exact solve for n={n}, m={G.number_of_edges()}.")

    solver = _new_solver()
    x = [Int(f"cx_{i}") for i in range(n)]
    p = [Int(f"cp_{i}") for i in range(n)]
    L = Int("cL")
    U = Int("cU")

    solver.add(x[0] == 0)
    if n >= 2:
        solver.add(x[1] >= 0)

    for i in range(n):
        solver.add(p[i] >= 1)
    solver.add(L >= 0)
    solver.add(U >= L)

    for i, j, is_edge in pairs:
        dij = p[i] + p[j] + _abs_z3(x[i] - x[j])
        if is_edge:
            solver.add(dij >= L, dij <= U)
        else:
            solver.add(Or(dij < L, dij > U))

    _log(verbose, "Caterpillar constraints added. Invoking Z3.")
    status = solver.check()
    _log(verbose, f"Caterpillar solver status: {status}")

    if status == sat:
        model = solver.model()
        interval = [
            model.evaluate(L, model_completion=True).as_long(),
            model.evaluate(U, model_completion=True).as_long(),
        ]
        xs = [model.evaluate(x[i], model_completion=True).as_long() for i in range(n)]
        ps = [model.evaluate(p[i], model_completion=True).as_long() for i in range(n)]
        distance_matrix = _build_caterpillar_distance_matrix(xs, ps)
        if not _check_interval_against_graph(pairs, distance_matrix, interval):
            raise RuntimeError("Internal error: caterpillar witness does not validate against the graph.")
        _log(verbose, f"Caterpillar witness found with interval={interval}.")
        _log(verbose, f"Caterpillar spine positions x={xs}")
        _log(verbose, f"Caterpillar pendant lengths p={ps}")
        return {
            "status": "sat-caterpillar",
            "interval": interval,
            "distance_matrix": distance_matrix,
            "note": None,
        }

    if status == unsat:
        _log(verbose, "Graph is not caterpillar-PCG under the exact encoding.")
        return {
            "status": "unsat",
            "interval": None,
            "distance_matrix": None,
            "note": None,
        }

    reason = solver.reason_unknown()
    _log(verbose, f"Caterpillar solver returned unknown: {reason}")
    raise RuntimeError(f"caterpillar solver returned unknown: {reason}")


def _make_general_distance_vars(n: int):
    dvars = {(i, j): Int(f"gd_{i}_{j}") for i in range(n) for j in range(i + 1, n)}

    def dij(i: int, j: int):
        if i == j:
            return IntVal(0)
        if i < j:
            return dvars[(i, j)]
        return dvars[(j, i)]

    return dvars, dij


def _add_general_metric_constraints(solver, dvars, dij, n: int, verbose: bool = False) -> None:
    _log(verbose, "Adding general metric constraints (positivity, triangles, four-point condition).")

    for var in dvars.values():
        solver.add(var >= 1)

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                dab = dij(a, b)
                dac = dij(a, c)
                dbc = dij(b, c)
                solver.add(dab <= dac + dbc)
                solver.add(dac <= dab + dbc)
                solver.add(dbc <= dab + dac)

    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    s1 = dij(a, b) + dij(c, d)
                    s2 = dij(a, c) + dij(b, d)
                    s3 = dij(a, d) + dij(b, c)
                    solver.add(
                        Or(
                            And(s1 == s2, s1 >= s3),
                            And(s1 == s3, s1 >= s2),
                            And(s2 == s3, s2 >= s1),
                        )
                    )


def _build_general_distance_matrix(model, dvars, n: int) -> List[List[int]]:
    dist = [[0] * n for _ in range(n)]
    for (i, j), var in dvars.items():
        value = model.evaluate(var, model_completion=True).as_long()
        dist[i][j] = value
        dist[j][i] = value
    return dist


def _solve_general_pcg_exact(G: nx.Graph, pairs: PairList, verbose: bool = False) -> JsonDict:
    n = G.number_of_nodes()
    _log(verbose, f"Starting general PCG exact solve for n={n}, m={G.number_of_edges()}.")

    solver = _new_solver()
    dvars, dij = _make_general_distance_vars(n)
    L = Int("gL")
    U = Int("gU")

    _add_general_metric_constraints(solver, dvars, dij, n, verbose=verbose)
    solver.add(L >= 0)
    solver.add(U >= L)

    for i, j, is_edge in pairs:
        d_ij = dij(i, j)
        if is_edge:
            solver.add(d_ij >= L, d_ij <= U)
        else:
            solver.add(Or(d_ij < L, d_ij > U))

    _log(verbose, "General PCG constraints added. Invoking Z3.")
    status = solver.check()
    _log(verbose, f"General solver status: {status}")

    if status == sat:
        model = solver.model()
        interval = [
            model.evaluate(L, model_completion=True).as_long(),
            model.evaluate(U, model_completion=True).as_long(),
        ]
        distance_matrix = _build_general_distance_matrix(model, dvars, n)
        if not _check_interval_against_graph(pairs, distance_matrix, interval):
            raise RuntimeError("Internal error: general-PCG witness does not validate against the graph.")
        _log(verbose, f"General PCG witness found with interval={interval}.")
        return {
            "status": "sat-general",
            "interval": interval,
            "distance_matrix": distance_matrix,
            "note": None,
        }

    if status == unsat:
        _log(verbose, "Graph is not a PCG under the exact general encoding.")
        return {
            "status": "unsat",
            "interval": None,
            "distance_matrix": None,
            "note": None,
        }

    reason = solver.reason_unknown()
    _log(verbose, f"General solver returned unknown: {reason}")
    raise RuntimeError(f"general solver returned unknown: {reason}")


def analyze_one_graph6(g6: str, verbose: bool = False) -> JsonDict:
    start_time = time.time()
    g6 = g6.strip()
    G = _graph_from_graph6(g6)
    pairs = _pair_data(G)

    base = {
        "n": G.number_of_nodes(),
        "graph6": g6,
        "mode": "exact",
        "edges": _edges_of_graph(G),
        "interval": None,
        "distance_matrix": None,
        "time_elapsed": None,
        "note": None,
    }

    _log(verbose, f"Analyzing graph6={g6!r} with n={base['n']} and m={G.number_of_edges()}.")

    phase_start = time.time()
    star = _solve_star_pcg_exact(G, pairs, verbose=verbose)
    _log(verbose, f"Star phase elapsed: {time.time() - phase_start:.4f}s")
    if star["status"] == "sat-star":
        elapsed = time.time() - start_time
        result = {**base, **star, "time_elapsed": elapsed}
        _log(verbose, f"Finished graph6={g6!r}: status={result['status']} in {elapsed:.4f}s.")
        return result

    phase_start = time.time()
    caterpillar = _solve_caterpillar_pcg_exact(G, pairs, verbose=verbose)
    _log(verbose, f"Caterpillar phase elapsed: {time.time() - phase_start:.4f}s")
    if caterpillar["status"] == "sat-caterpillar":
        elapsed = time.time() - start_time
        result = {**base, **caterpillar, "time_elapsed": elapsed}
        _log(verbose, f"Finished graph6={g6!r}: status={result['status']} in {elapsed:.4f}s.")
        return result

    phase_start = time.time()
    general = _solve_general_pcg_exact(G, pairs, verbose=verbose)
    _log(verbose, f"General phase elapsed: {time.time() - phase_start:.4f}s")

    elapsed = time.time() - start_time
    result = {**base, **general, "time_elapsed": elapsed}
    _log(verbose, f"Finished graph6={g6!r}: status={result['status']} in {elapsed:.4f}s.")
    return result


def analyze_graph6_list(graph6_list: Sequence[str], verbose: bool = False) -> List[JsonDict]:
    _log(verbose, f"Analyzing list of {len(graph6_list)} graph(s).")
    return [analyze_one_graph6(g6, verbose=verbose) for g6 in graph6_list]


def analyze_graph6_file(path: Union[str, Path], verbose: bool = False) -> List[JsonDict]:
    graph6_list = _read_graph6_file(path, verbose=verbose)
    return analyze_graph6_list(graph6_list, verbose=verbose)


def analyze(source: Graph6Source, verbose: bool = False) -> List[JsonDict]:
    """Analyze one or more graph6 graphs.

    Parameters
    ----------
    source:
        One of:
        - a single graph6 string,
        - a list/tuple of graph6 strings,
        - a filename / Path containing one graph6 string per line.
    verbose:
        If True, print extensive progress logs.

    Returns
    -------
    list[dict]
        One JSON-serializable result dict per graph.
    """
    graph6_list = _normalize_source(source, verbose=verbose)
    return analyze_graph6_list(graph6_list, verbose=verbose)


def results_to_jsonable(results: Sequence[Dict]) -> List[JsonDict]:
    return list(results)
