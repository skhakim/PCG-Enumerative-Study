#!/usr/bin/env python3
"""
Multiprocessing filter for minimal non-PCGs.

For n=10, default input/output:

    smaller non-PCGs: results/vert_9/non_pcg_9.jsonl
    current non-PCGs: results/vert_10/non_pcg_10.jsonl

Outputs:
    results/vert_10/minimal_non_pcg_10.jsonl
    results/vert_10/nonminimal_non_pcg_10.jsonl
    results/vert_10/minimal_non_pcg_10_summary.txt

Criterion:
    A non-PCG G on n vertices is minimal iff every vertex-deleted
    induced subgraph G-v is PCG.

Since PCGs are hereditary, for a current n-vertex non-PCG G, it is enough to
test whether any G-v appears in the complete catalogue of (n-1)-vertex non-PCGs.

This version parallelizes over chunks of current non-PCGs. The parent process
loads/canonicalizes the (n-1)-vertex non-PCG oracle once, forks workers, and
workers inherit the oracle set copy-on-write on Linux.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import networkx as nx


# ---------------------------------------------------------------------
# Globals inherited by forked workers.
# ---------------------------------------------------------------------

GLOBAL_SMALLER_NONPCG_SET = set()
GLOBAL_LABELG_PATH = None
GLOBAL_N = None
GLOBAL_LABELG_BATCH_SIZE = None


# ---------------------------------------------------------------------
# Basic utilities.
# ---------------------------------------------------------------------

def find_labelg(user_path: Optional[str] = None) -> str:
    if user_path:
        found = shutil.which(user_path)
        if found is None:
            raise FileNotFoundError("Could not find labelg executable: %s" % user_path)
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
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON in %s at line %d" % (path, line_no)) from exc


def read_graph6_list_from_jsonl(path: Path) -> List[str]:
    graph6s = []
    for rec in read_jsonl_records(path):
        if "graph6" not in rec:
            raise KeyError("Missing 'graph6' field in record from %s" % path)
        graph6s.append(rec["graph6"].strip())
    return graph6s


def chunked_records(path: Path, chunk_size: int) -> Iterator[List[Dict[str, Any]]]:
    chunk = []
    for rec in read_jsonl_records(path):
        if "graph6" not in rec:
            raise KeyError("Missing 'graph6' field in record from %s" % path)
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def canonicalize_batch(
    graph6s: Sequence[str],
    labelg_path: str,
    batch_label: str = "",
) -> List[str]:
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
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "labelg failed while canonicalizing batch %s.\n"
            "Command: %s -q\nstderr:\n%s"
            % (batch_label, labelg_path, exc.stderr)
        ) from exc

    out = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith(">")
    ]

    if len(out) != len(graph6s):
        raise RuntimeError(
            "labelg output count mismatch in batch %s: input=%d, output=%d.\n"
            "stderr:\n%s"
            % (batch_label, len(graph6s), len(out), proc.stderr[:1000])
        )

    return out


def canonicalize_large_list(
    graph6s: Sequence[str],
    labelg_path: str,
    batch_size: int,
    desc: str,
) -> List[str]:
    out = []
    total = len(graph6s)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = graph6s[start:end]
        out.extend(
            canonicalize_batch(
                batch,
                labelg_path=labelg_path,
                batch_label="%s %d:%d" % (desc, start, end),
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
    return "%02d:%02d:%02d" % (h, m, s)


# ---------------------------------------------------------------------
# Worker.
# ---------------------------------------------------------------------

def init_worker(
    smaller_nonpcg_set,
    labelg_path: str,
    n: int,
    labelg_batch_size: int,
) -> None:
    global GLOBAL_SMALLER_NONPCG_SET
    global GLOBAL_LABELG_PATH
    global GLOBAL_N
    global GLOBAL_LABELG_BATCH_SIZE

    GLOBAL_SMALLER_NONPCG_SET = smaller_nonpcg_set
    GLOBAL_LABELG_PATH = labelg_path
    GLOBAL_N = n
    GLOBAL_LABELG_BATCH_SIZE = labelg_batch_size


def process_chunk_worker(records: List[Dict[str, Any]]) -> Tuple[int, int, int, List[str], List[str]]:
    """
    Process one chunk of current non-PCGs.

    Returns:
        total_in_chunk,
        minimal_count,
        nonminimal_count,
        minimal_json_lines,
        nonminimal_json_lines
    """
    n = GLOBAL_N
    labelg_path = GLOBAL_LABELG_PATH
    labelg_batch_size = GLOBAL_LABELG_BATCH_SIZE
    smaller_nonpcg_set = GLOBAL_SMALLER_NONPCG_SET

    raw_deleted = []
    owner = []

    for idx, rec in enumerate(records):
        g6 = rec["graph6"].strip()
        dels = vertex_deleted_graph6s(g6)

        if len(dels) != n:
            raise RuntimeError(
                "Expected %d vertex-deleted subgraphs, got %d for graph6=%s"
                % (n, len(dels), g6)
            )

        for h in dels:
            raw_deleted.append(h)
            owner.append(idx)

    can_deleted = canonicalize_large_list(
        raw_deleted,
        labelg_path=labelg_path,
        batch_size=labelg_batch_size,
        desc="worker-deleted-subgraphs",
    )

    is_nonminimal = [False] * len(records)
    blocker = [None] * len(records)

    for h_can, idx in zip(can_deleted, owner):
        if h_can in smaller_nonpcg_set:
            is_nonminimal[idx] = True
            if blocker[idx] is None:
                blocker[idx] = h_can

    minimal_lines = []
    nonminimal_lines = []

    minimal_count = 0
    nonminimal_count = 0

    for idx, rec in enumerate(records):
        out_rec = dict(rec)

        if is_nonminimal[idx]:
            nonminimal_count += 1
            out_rec["minimality_status"] = "nonminimal"
            out_rec["nonpcg_induced_subgraph_on_n_minus_1"] = blocker[idx]
            nonminimal_lines.append(json.dumps(out_rec, separators=(",", ":")) + "\n")
        else:
            minimal_count += 1
            out_rec["minimality_status"] = "minimal"
            minimal_lines.append(json.dumps(out_rec, separators=(",", ":")) + "\n")

    return len(records), minimal_count, nonminimal_count, minimal_lines, nonminimal_lines


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiprocessing filter for minimal non-PCGs."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Vertex count of current non-PCG file. Default: 10.",
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
        help="Output JSONL for minimal non-PCGs.",
    )
    parser.add_argument(
        "--nonminimal-out",
        type=Path,
        default=None,
        help="Output JSONL for nonminimal non-PCGs.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Output summary file.",
    )
    parser.add_argument(
        "--labelg",
        type=str,
        default=None,
        help="Path/name of nauty labelg. Auto-detects 'nauty-labelg' or 'labelg'.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes. On ibm8335, try 120--150.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of current graphs per worker task. Default: 1000.",
    )
    parser.add_argument(
        "--labelg-batch-size",
        type=int,
        default=20000,
        help="Number of graph6 strings per labelg call. Default: 20000.",
    )
    parser.add_argument(
        "--oracle-batch-size",
        type=int,
        default=100000,
        help="Batch size for canonicalizing the smaller non-PCG oracle.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every this many processed graphs.",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Optional number of chunks to process for pilot runs.",
    )

    args = parser.parse_args()

    n = args.n
    smaller = args.smaller or Path("results/vert_%d/non_pcg_%d.jsonl" % (n - 1, n - 1))
    current = args.current or Path("results/vert_%d/non_pcg_%d.jsonl" % (n, n))
    minimal_out = args.minimal_out or Path("results/vert_%d/minimal_non_pcg_%d.jsonl" % (n, n))
    nonminimal_out = args.nonminimal_out or Path("results/vert_%d/nonminimal_non_pcg_%d.jsonl" % (n, n))
    summary = args.summary or Path("results/vert_%d/minimal_non_pcg_%d_summary.txt" % (n, n))

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.labelg_batch_size < 1:
        raise ValueError("--labelg-batch-size must be >= 1")

    for path in (smaller, current):
        if not path.exists():
            raise FileNotFoundError("Input file not found: %s" % path)

    labelg_path = find_labelg(args.labelg)

    minimal_out.parent.mkdir(parents=True, exist_ok=True)
    nonminimal_out.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    # Clear output files.
    minimal_out.write_text("", encoding="utf-8")
    nonminimal_out.write_text("", encoding="utf-8")

    start_time = time.perf_counter()

    print("=" * 72, file=sys.stderr)
    print("[config]", file=sys.stderr)
    print("n:                  %d" % n, file=sys.stderr)
    print("smaller:            %s" % smaller, file=sys.stderr)
    print("current:            %s" % current, file=sys.stderr)
    print("minimal_out:        %s" % minimal_out, file=sys.stderr)
    print("nonminimal_out:     %s" % nonminimal_out, file=sys.stderr)
    print("summary:            %s" % summary, file=sys.stderr)
    print("labelg:             %s" % labelg_path, file=sys.stderr)
    print("jobs:               %d" % args.jobs, file=sys.stderr)
    print("chunk_size:         %d" % args.chunk_size, file=sys.stderr)
    print("labelg_batch_size:  %d" % args.labelg_batch_size, file=sys.stderr)
    print("oracle_batch_size:  %d" % args.oracle_batch_size, file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    print("[info] reading smaller non-PCGs...", file=sys.stderr)
    smaller_graph6s = read_graph6_list_from_jsonl(smaller)
    print("[info] loaded %d smaller non-PCGs" % len(smaller_graph6s), file=sys.stderr)

    print("[info] canonicalizing smaller non-PCG oracle...", file=sys.stderr)
    smaller_can = canonicalize_large_list(
        smaller_graph6s,
        labelg_path=labelg_path,
        batch_size=args.oracle_batch_size,
        desc="nonpcg_%d" % (n - 1),
    )
    smaller_nonpcg_set = set(smaller_can)
    print(
        "[info] canonical smaller non-PCG set size: %d" % len(smaller_nonpcg_set),
        file=sys.stderr,
    )

    total = 0
    minimal = 0
    nonminimal = 0
    next_progress = args.progress_every

    chunk_iter = chunked_records(current, args.chunk_size)

    if args.limit_chunks is not None:
        def limited_chunks():
            for k, chunk in enumerate(chunk_iter):
                if k >= args.limit_chunks:
                    break
                yield chunk
        work_iter = limited_chunks()
    else:
        work_iter = chunk_iter

    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp

    print("[info] starting multiprocessing workers...", file=sys.stderr, flush=True)

    with minimal_out.open("a", encoding="utf-8") as f_min, \
         nonminimal_out.open("a", encoding="utf-8") as f_nonmin:

        if args.jobs == 1:
            init_worker(smaller_nonpcg_set, labelg_path, n, args.labelg_batch_size)
            result_iter = map(process_chunk_worker, work_iter)
        else:
            pool = ctx.Pool(
                processes=args.jobs,
                initializer=init_worker,
                initargs=(smaller_nonpcg_set, labelg_path, n, args.labelg_batch_size),
            )
            result_iter = pool.imap_unordered(process_chunk_worker, work_iter, chunksize=1)

        try:
            for chunk_total, chunk_minimal, chunk_nonminimal, min_lines, nonmin_lines in result_iter:
                total += chunk_total
                minimal += chunk_minimal
                nonminimal += chunk_nonminimal

                f_min.writelines(min_lines)
                f_nonmin.writelines(nonmin_lines)

                if total >= next_progress:
                    f_min.flush()
                    f_nonmin.flush()

                    elapsed = time.perf_counter() - start_time
                    rate = total / elapsed if elapsed > 0 else 0.0
                    print(
                        "[progress] processed=%s, minimal=%s, nonminimal=%s, "
                        "elapsed=%s, rate=%.2f/s"
                        % (
                            format(total, ","),
                            format(minimal, ","),
                            format(nonminimal, ","),
                            format_hms(elapsed),
                            rate,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    while next_progress <= total:
                        next_progress += args.progress_every

        finally:
            if args.jobs != 1:
                pool.close()
                pool.join()

    elapsed = time.perf_counter() - start_time

    with summary.open("w", encoding="utf-8") as f:
        f.write("n: %d\n" % n)
        f.write("smaller_nonpcg_file: %s\n" % smaller)
        f.write("current_nonpcg_file: %s\n" % current)
        f.write("minimal_output_file: %s\n" % minimal_out)
        f.write("nonminimal_output_file: %s\n" % nonminimal_out)
        f.write("total_current_nonpcg: %d\n" % total)
        f.write("minimal_nonpcg: %d\n" % minimal)
        f.write("nonminimal_nonpcg: %d\n" % nonminimal)
        f.write("elapsed_seconds: %.6f\n" % elapsed)
        f.write("elapsed_hms: %s\n" % format_hms(elapsed))
        f.write("jobs: %d\n" % args.jobs)
        f.write("chunk_size: %d\n" % args.chunk_size)
        f.write("labelg_path: %s\n" % labelg_path)
        f.write("labelg_batch_size: %d\n" % args.labelg_batch_size)
        f.write("oracle_batch_size: %d\n" % args.oracle_batch_size)

    print("[done]", file=sys.stderr)
    print("total_current_nonpcg: %s" % format(total, ","), file=sys.stderr)
    print("minimal_nonpcg:       %s" % format(minimal, ","), file=sys.stderr)
    print("nonminimal_nonpcg:    %s" % format(nonminimal, ","), file=sys.stderr)
    print("elapsed:              %s" % format_hms(elapsed), file=sys.stderr)
    print("summary:              %s" % summary, file=sys.stderr)


if __name__ == "__main__":
    main()
