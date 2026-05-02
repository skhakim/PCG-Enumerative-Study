#!/usr/bin/env python3
"""
Filter minimal non-PCGs from a non-PCG JSONL file.

For n=10, the default behavior is:

    smaller non-PCGs: results/vert_9/non_pcg_9.jsonl
    current non-PCGs: results/vert_10/non_pcg_10.jsonl
    output minimal:   results/vert_10/minimal_non_pcg_10.jsonl

A graph G on n vertices is minimal non-PCG iff:
    G is non-PCG, and
    every vertex-deleted induced subgraph G-v is PCG.

Since PCGs are hereditary, for a non-PCG G it is enough to check whether
any G-v appears in the complete list of non-PCGs on n-1 vertices.

This script uses nauty labelg to canonicalize induced subgraphs in batches.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import networkx as nx


def find_labelg(user_path: str | None = None) -> str:
    if user_path:
        if shutil.which(user_path) is None:
            raise FileNotFoundError(f"Could not find labelg executable: {user_path}")
        return user_path

    for candidate in ("nauty-labelg", "labelg"):
        found = shutil.which(candidate)
        if found:
            return candidate

    raise FileNotFoundError(
        "Could not find nauty labelg. Tried 'nauty-labelg' and 'labelg'. "
        "Pass it explicitly using --labelg /path/to/labelg."
    )


def read_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from e


def read_graph6_list_from_jsonl(path: Path) -> List[str]:
    graph6s = []
    for rec in read_jsonl_records(path):
        if "graph6" not in rec:
            raise KeyError(f"Missing 'graph6' field in record from {path}")
        graph6s.append(rec["graph6"].strip())
    return graph6s


def canonicalize_batch(
    graph6s: List[str],
    labelg_path: str,
    batch_label: str = "",
) -> List[str]:
    """
    Canonicalize a list of graph6 strings using nauty labelg.

    Returns one canonical graph6 string per input graph.
    """
    if not graph6s:
        return []

    input_text = "\n".join(g.strip() for g in graph6s) + "\n"

    try:
        proc = subprocess.run(
            [labelg_path, "-q"],
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"labelg failed while canonicalizing batch {batch_label}.\n"
            f"Command: {labelg_path} -q\n"
            f"stderr:\n{e.stderr}"
        ) from e

    out = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith(">")
    ]

    if len(out) != len(graph6s):
        raise RuntimeError(
            f"labelg output count mismatch in batch {batch_label}: "
            f"input={len(graph6s)}, output={len(out)}.\n"
            f"stderr:\n{proc.stderr[:1000]}"
        )

    return out


def canonicalize_large_list(
    graph6s: List[str],
    labelg_path: str,
    batch_size: int,
    desc: str,
) -> List[str]:
    out: List[str] = []
    total = len(graph6s)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = graph6s[start:end]
        out.extend(
            canonicalize_batch(
                batch,
                labelg_path=labelg_path,
                batch_label=f"{desc} {start}:{end}",
            )
        )

    return out


def graph_from_graph6(g6: str) -> nx.Graph:
    G = nx.from_graph6_bytes(g6.strip().encode("ascii"))
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def to_raw_graph6(G: nx.Graph) -> str:
    H = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return nx.to_graph6_bytes(H, header=False).decode("ascii").strip()


def vertex_deleted_graph6s(g6: str) -> List[str]:
    G = graph_from_graph6(g6)
    nodes = list(G.nodes())

    deleted = []
    for v in nodes:
        keep = [u for u in nodes if u != v]
        H = G.subgraph(keep).copy()
        deleted.append(to_raw_graph6(H))

    return deleted


def format_hms(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def filter_minimal_nonpcg(
    n: int,
    smaller_path: Path,
    current_path: Path,
    minimal_out_path: Path,
    nonminimal_out_path: Path,
    summary_path: Path,
    labelg_path: str,
    canonical_batch_size: int = 50000,
    graph_chunk_size: int = 2000,
) -> None:
    start_time = time.perf_counter()

    print(f"[info] n = {n}")
    print(f"[info] smaller non-PCGs: {smaller_path}")
    print(f"[info] current non-PCGs: {current_path}")
    print(f"[info] labelg: {labelg_path}")
    print("[info] reading smaller non-PCGs...")

    smaller_graph6s = read_graph6_list_from_jsonl(smaller_path)
    print(f"[info] loaded {len(smaller_graph6s)} smaller non-PCGs")

    print("[info] canonicalizing smaller non-PCGs...")
    smaller_can = canonicalize_large_list(
        smaller_graph6s,
        labelg_path=labelg_path,
        batch_size=canonical_batch_size,
        desc=f"nonpcg_{n-1}",
    )
    smaller_nonpcg_set = set(smaller_can)

    print(f"[info] canonical smaller non-PCG set size: {len(smaller_nonpcg_set)}")

    minimal_out_path.parent.mkdir(parents=True, exist_ok=True)
    nonminimal_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    minimal = 0
    nonminimal = 0

    chunk_records: List[Dict[str, Any]] = []

    def process_chunk(records: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not records:
            return 0, 0

        raw_deleted: List[str] = []
        owner: List[int] = []

        for idx, rec in enumerate(records):
            g6 = rec["graph6"].strip()
            dels = vertex_deleted_graph6s(g6)
            if len(dels) != n:
                raise RuntimeError(
                    f"Expected {n} vertex-deleted subgraphs, got {len(dels)} "
                    f"for graph6={g6}"
                )
            for h in dels:
                raw_deleted.append(h)
                owner.append(idx)

        can_deleted = canonicalize_large_list(
            raw_deleted,
            labelg_path=labelg_path,
            batch_size=canonical_batch_size,
            desc=f"deleted-subgraphs-{total}",
        )

        is_nonminimal = [False] * len(records)
        blocker = [None] * len(records)

        for h_can, idx in zip(can_deleted, owner):
            if h_can in smaller_nonpcg_set:
                is_nonminimal[idx] = True
                if blocker[idx] is None:
                    blocker[idx] = h_can

        local_minimal = 0
        local_nonminimal = 0

        with minimal_out_path.open("a", encoding="utf-8") as f_min, \
             nonminimal_out_path.open("a", encoding="utf-8") as f_nonmin:

            for idx, rec in enumerate(records):
                if is_nonminimal[idx]:
                    local_nonminimal += 1
                    out_rec = dict(rec)
                    out_rec["minimality_status"] = "nonminimal"
                    out_rec["nonpcg_induced_subgraph_on_n_minus_1"] = blocker[idx]
                    f_nonmin.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                else:
                    local_minimal += 1
                    out_rec = dict(rec)
                    out_rec["minimality_status"] = "minimal"
                    f_min.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        return local_minimal, local_nonminimal

    # Clear previous output files.
    minimal_out_path.write_text("", encoding="utf-8")
    nonminimal_out_path.write_text("", encoding="utf-8")

    print("[info] processing current non-PCGs...")

    for rec in read_jsonl_records(current_path):
        if "graph6" not in rec:
            raise KeyError(f"Missing 'graph6' field in record from {current_path}")

        chunk_records.append(rec)

        if len(chunk_records) >= graph_chunk_size:
            cm, cnm = process_chunk(chunk_records)
            total += len(chunk_records)
            minimal += cm
            nonminimal += cnm
            elapsed = time.perf_counter() - start_time
            print(
                f"[progress] processed={total:,}, minimal={minimal:,}, "
                f"nonminimal={nonminimal:,}, elapsed={format_hms(elapsed)}",
                flush=True,
            )
            chunk_records = []

    if chunk_records:
        cm, cnm = process_chunk(chunk_records)
        total += len(chunk_records)
        minimal += cm
        nonminimal += cnm
        elapsed = time.perf_counter() - start_time
        print(
            f"[progress] processed={total:,}, minimal={minimal:,}, "
            f"nonminimal={nonminimal:,}, elapsed={format_hms(elapsed)}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"n: {n}\n")
        f.write(f"smaller_nonpcg_file: {smaller_path}\n")
        f.write(f"current_nonpcg_file: {current_path}\n")
        f.write(f"minimal_output_file: {minimal_out_path}\n")
        f.write(f"nonminimal_output_file: {nonminimal_out_path}\n")
        f.write(f"total_current_nonpcg: {total}\n")
        f.write(f"minimal_nonpcg: {minimal}\n")
        f.write(f"nonminimal_nonpcg: {nonminimal}\n")
        f.write(f"elapsed_seconds: {elapsed:.6f}\n")
        f.write(f"elapsed_hms: {format_hms(elapsed)}\n")
        f.write(f"labelg_path: {labelg_path}\n")
        f.write(f"canonical_batch_size: {canonical_batch_size}\n")
        f.write(f"graph_chunk_size: {graph_chunk_size}\n")

    print("[done]")
    print(f"total_current_nonpcg: {total:,}")
    print(f"minimal_nonpcg:       {minimal:,}")
    print(f"nonminimal_nonpcg:    {nonminimal:,}")
    print(f"elapsed:              {format_hms(elapsed)}")
    print(f"summary:              {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter minimal non-PCGs using vertex-deleted induced subgraphs."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Vertex count of the current non-PCG file. Default: 10.",
    )
    parser.add_argument(
        "--smaller",
        type=Path,
        default=None,
        help="Path to non_pcg_{n-1}.jsonl. Default: results/vert_{n-1}/non_pcg_{n-1}.jsonl",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="Path to non_pcg_n.jsonl. Default: results/vert_n/non_pcg_n.jsonl",
    )
    parser.add_argument(
        "--minimal-out",
        type=Path,
        default=None,
        help="Output JSONL for minimal non-PCGs. Default: results/vert_n/minimal_non_pcg_n.jsonl",
    )
    parser.add_argument(
        "--nonminimal-out",
        type=Path,
        default=None,
        help="Output JSONL for nonminimal non-PCGs. Default: results/vert_n/nonminimal_non_pcg_n.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Output summary file. Default: results/vert_n/minimal_non_pcg_n_summary.txt",
    )
    parser.add_argument(
        "--labelg",
        type=str,
        default=None,
        help="Path/name of nauty labelg. Auto-detects 'nauty-labelg' or 'labelg'.",
    )
    parser.add_argument(
        "--canonical-batch-size",
        type=int,
        default=50000,
        help="Number of graph6 strings to canonicalize per labelg call. Default: 50000.",
    )
    parser.add_argument(
        "--graph-chunk-size",
        type=int,
        default=2000,
        help="Number of current graphs processed per chunk. Default: 2000.",
    )

    args = parser.parse_args()

    n = args.n
    smaller = args.smaller or Path(f"results/vert_{n-1}/non_pcg_{n-1}.jsonl")
    current = args.current or Path(f"results/vert_{n}/non_pcg_{n}.jsonl")
    minimal_out = args.minimal_out or Path(f"results/vert_{n}/minimal_non_pcg_{n}.jsonl")
    nonminimal_out = args.nonminimal_out or Path(f"results/vert_{n}/nonminimal_non_pcg_{n}.jsonl")
    summary = args.summary or Path(f"results/vert_{n}/minimal_non_pcg_{n}_summary.txt")
    labelg_path = find_labelg(args.labelg)

    for path in (smaller, current):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    filter_minimal_nonpcg(
        n=n,
        smaller_path=smaller,
        current_path=current,
        minimal_out_path=minimal_out,
        nonminimal_out_path=nonminimal_out,
        summary_path=summary,
        labelg_path=labelg_path,
        canonical_batch_size=args.canonical_batch_size,
        graph_chunk_size=args.graph_chunk_size,
    )


if __name__ == "__main__":
    main()
