#!/usr/bin/env python3
"""
Parallel search for non-2-AND-PCGs using a negative PCG oracle.

Context
-------
For n=10, we have a complete list of non-PCGs:

    results/vert_10/non_pcg_10.jsonl

Therefore, for any connected n-vertex graph H:

    H is PCG  iff  canonical(H) is NOT in the non-PCG oracle set.

For a target non-PCG G, we test whether G is 2-AND-PCG.

Definition
----------
G is 2-AND-PCG iff there exist two PCG supergraphs H1 and H2 of G
on the same vertex set such that

    G = H1 ∩ H2.

Equivalently, there exist two disjoint added-edge sets A and B such that:

    H1 = G ∪ A is PCG,
    H2 = G ∪ B is PCG,
    A ∩ B = empty.

This script searches added-edge masks in increasing size and uses the
non-PCG oracle to decide whether each candidate supergraph is PCG.

Important
---------
This script has three statuses:

    2-and-pcg
        A witness pair was found. By default, not written to disk.

    proven-non-2and
        Exhaustive search completed and no witness pair exists.
        This is only a proof when --max-add-size -1 and --timeout-per-graph -1.

    unresolved
        Search stopped due to timeout or max-add-size bound.

Default usage
-------------
Pilot:

    head -n 100 results/vert_10/non_pcg_10.jsonl > results/vert_10/non_pcg_10_first100.jsonl

    python3 check_non2and_by_negative_oracle_parallel.py \
        --n 10 \
        --targets results/vert_10/non_pcg_10_first100.jsonl \
        --max-add-size 5 \
        --timeout-per-graph 60 \
        --jobs 32 \
        --progress-every 10

Full bounded run:

    tmux new -s non2and10

    python3 check_non2and_by_negative_oracle_parallel.py \
        --n 10 \
        --max-add-size 5 \
        --timeout-per-graph 60 \
        --jobs 120 \
        --progress-every 1000

Exact run on a smaller hard residue:

    python3 check_non2and_by_negative_oracle_parallel.py \
        --n 10 \
        --targets results/vert_10/unresolved_2and_pcg_10.jsonl \
        --max-add-size -1 \
        --timeout-per-graph -1 \
        --jobs 120
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# Global state inherited by forked workers.
# ---------------------------------------------------------------------

GLOBAL_ORACLE_NONPCG: set[str] = set()
GLOBAL_CONFIG: dict = {}


# ---------------------------------------------------------------------
# Basic graph-mask utilities.
# ---------------------------------------------------------------------

def build_edge_tables(n: int) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_index = {e: idx for idx, e in enumerate(edge_list)}
    return edge_list, edge_to_index


def mask_from_edges(
    edges: Sequence[Sequence[int]],
    edge_to_index: Dict[Tuple[int, int], int],
) -> int:
    mask = 0
    for a, b in edges:
        if a > b:
            a, b = b, a
        mask |= 1 << edge_to_index[(a, b)]
    return mask


def edges_from_mask(mask: int, edge_list: Sequence[Tuple[int, int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for idx, (i, j) in enumerate(edge_list):
        if (mask >> idx) & 1:
            out.append([i, j])
    return out


def iter_bits(mask: int) -> Iterator[int]:
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb



def popcount(x: int) -> int:
    return bin(x).count("1")


def graph6_from_mask(
    mask: int,
    n: int,
    edge_to_index: Dict[Tuple[int, int], int],
) -> str:
    """
    Encode a simple graph as graph6 for n <= 62.

    graph6 bit order:
        (0,1),
        (0,2), (1,2),
        (0,3), (1,3), (2,3),
        ...
        (0,n-1), ..., (n-2,n-1).
    """
    if n > 62:
        raise ValueError("This simple graph6 encoder supports only n <= 62.")

    bits: List[int] = []

    for j in range(1, n):
        for i in range(j):
            idx = edge_to_index[(i, j)]
            bits.append((mask >> idx) & 1)

    while len(bits) % 6 != 0:
        bits.append(0)

    chars = [chr(n + 63)]

    for k in range(0, len(bits), 6):
        val = 0
        for b in bits[k:k + 6]:
            val = (val << 1) | b
        chars.append(chr(val + 63))

    return "".join(chars)


def combinations_as_masks(items: List[int], k: int) -> Iterator[int]:
    for comb in itertools.combinations(items, k):
        m = 0
        for idx in comb:
            m |= 1 << idx
        yield m


def format_hms(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------
# JSONL utilities.
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def count_jsonl_records(path: Path, n: Optional[int] = None) -> int:
    count = 0
    for rec in read_jsonl(path):
        if n is not None and rec.get("n") is not None and rec["n"] != n:
            continue
        count += 1
    return count


def iter_target_records(
    path: Path,
    n: int,
    limit: Optional[int] = None,
) -> Iterator[Tuple[int, dict]]:
    yielded = 0

    for idx, rec in enumerate(read_jsonl(path)):
        if rec.get("n") is not None and rec["n"] != n:
            continue

        if "edges" not in rec:
            raise KeyError(
                f"Target record at index {idx} has no 'edges' field. "
                "This script expects JSONL records from your checker."
            )

        yield idx, rec
        yielded += 1

        if limit is not None and yielded >= limit:
            break


# ---------------------------------------------------------------------
# nauty labelg canonicalization.
# ---------------------------------------------------------------------

def find_labelg(user_path: Optional[str]) -> str:
    if user_path:
        if shutil.which(user_path) is None:
            raise FileNotFoundError(f"Could not find labelg executable: {user_path}")
        return user_path

    for cand in ("nauty-labelg", "labelg"):
        if shutil.which(cand):
            return cand

    raise FileNotFoundError(
        "Could not find nauty labelg. Tried 'nauty-labelg' and 'labelg'. "
        "Use --labelg /path/to/labelg."
    )


def canonicalize_batch(graph6s: List[str], labelg_path: str) -> List[str]:
    if not graph6s:
        return []

    proc = subprocess.run(
        [labelg_path, "-q"],
        input="\n".join(g.strip() for g in graph6s) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"labelg failed with return code {proc.returncode}\n"
            f"stderr:\n{proc.stderr}"
        )

    out = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith(">")
    ]

    if len(out) != len(graph6s):
        raise RuntimeError(
            f"labelg output count mismatch: input={len(graph6s)}, output={len(out)}\n"
            f"stderr:\n{proc.stderr[:1000]}"
        )

    return out


def load_oracle_nonpcg_set(
    path: Path,
    n: int,
    edge_to_index: Dict[Tuple[int, int], int],
    labelg_path: str,
    canonical_batch_size: int,
    trust_oracle_canonical: bool = False,
) -> set[str]:
    """
    Load the complete non-PCG oracle.

    If trust_oracle_canonical=False, every oracle graph is canonicalized using
    labelg. This is safer.

    If trust_oracle_canonical=True, graph6 strings are inserted directly.
    Use this only if you are certain the graph6 strings match labelg's
    canonical output.
    """
    print(f"[info] loading non-PCG oracle from {path}", file=sys.stderr)

    raw: List[str] = []
    total = 0

    for rec in read_jsonl(path):
        if rec.get("n") is not None and rec["n"] != n:
            continue

        if "graph6" in rec:
            g6 = rec["graph6"].strip()
        elif "edges" in rec:
            mask = mask_from_edges(rec["edges"], edge_to_index)
            g6 = graph6_from_mask(mask, n, edge_to_index)
        else:
            raise KeyError("Oracle record has neither 'graph6' nor 'edges'.")

        raw.append(g6)
        total += 1

    print(f"[info] oracle raw records loaded: {total:,}", file=sys.stderr)

    if trust_oracle_canonical:
        oracle = set(raw)
        print(
            f"[info] inserted oracle graph6 strings directly; unique={len(oracle):,}",
            file=sys.stderr,
        )
        return oracle

    oracle: set[str] = set()

    for start in range(0, len(raw), canonical_batch_size):
        end = min(start + canonical_batch_size, len(raw))
        can = canonicalize_batch(raw[start:end], labelg_path)
        oracle.update(can)

        print(
            f"[info] canonicalized oracle {end:,}/{len(raw):,}; "
            f"unique={len(oracle):,}",
            file=sys.stderr,
            flush=True,
        )

    return oracle


# ---------------------------------------------------------------------
# 2-AND search logic.
# ---------------------------------------------------------------------

def has_disjoint_partner(mask: int, good_masks: List[int]) -> Optional[int]:
    """
    Return an old good mask disjoint from mask, if one exists.
    """
    for old in good_masks:
        if (old & mask) == 0:
            return old
    return None


def process_candidate_batch(
    *,
    batch_added_masks: List[int],
    batch_graph6: List[str],
    good_masks: List[int],
    oracle_nonpcg: set[str],
    labelg_path: str,
    edge_list: Sequence[Tuple[int, int]],
) -> Tuple[Optional[Tuple[int, int, int]], int, int]:
    """
    Canonicalize a candidate batch and test PCG-ness using negative oracle.

    Returns:
        witness_pair:
            None if no witness pair found;
            otherwise (old_mask, new_mask, position_in_batch).

        candidates_tested_delta:
            number of candidates tested in this batch.

        pcg_supergraphs_delta:
            number of candidate supergraphs found to be PCG.
    """
    can_batch = canonicalize_batch(batch_graph6, labelg_path)

    candidates_tested_delta = 0
    pcg_supergraphs_delta = 0

    for pos, (am, can_h) in enumerate(zip(batch_added_masks, can_batch)):
        candidates_tested_delta += 1

        # Negative oracle:
        # H is PCG iff canonical(H) is not in the non-PCG oracle.
        if can_h not in oracle_nonpcg:
            pcg_supergraphs_delta += 1

            partner = has_disjoint_partner(am, good_masks)
            if partner is not None:
                return (partner, am, pos), candidates_tested_delta, pcg_supergraphs_delta

            good_masks.append(am)

    return None, candidates_tested_delta, pcg_supergraphs_delta


def test_target_2and_by_negative_oracle(
    *,
    target_index: int,
    rec: dict,
    n: int,
    edge_list: Sequence[Tuple[int, int]],
    edge_to_index: Dict[Tuple[int, int], int],
    all_edge_mask: int,
    oracle_nonpcg: set[str],
    labelg_path: str,
    max_add_size: Optional[int],
    timeout_per_graph: Optional[float],
    candidate_batch_size: int,
    debug_k: bool,
) -> dict:
    """
    Test one target graph.

    Returns a result dict with status_2and in:
        "2-and-pcg"
        "proven-non-2and"
        "unresolved"
    """
    t0 = time.perf_counter()

    g6 = rec.get("graph6")
    gmask = mask_from_edges(rec["edges"], edge_to_index)

    missing_mask = all_edge_mask & ~gmask
    missing_edges = list(iter_bits(missing_mask))
    missing_edge_count = len(missing_edges)
    edge_count = popcount(gmask)

    if max_add_size is None:
        effective_max_k = missing_edge_count
    else:
        effective_max_k = min(max_add_size, missing_edge_count)

    good_masks: List[int] = []

    candidates_tested = 0
    pcg_supergraphs_found = 0

    def timed_out() -> bool:
        return (
            timeout_per_graph is not None
            and (time.perf_counter() - t0) >= timeout_per_graph
        )

    for k in range(1, effective_max_k + 1):
        batch_added_masks: List[int] = []
        batch_graph6: List[str] = []

        for add_mask in combinations_as_masks(missing_edges, k):
            if timed_out():
                elapsed = time.perf_counter() - t0
                return {
                    "target_index": target_index,
                    "graph6": g6,
                    "n": n,
                    "status_2and": "unresolved",
                    "reason": "timeout",
                    "edge_count": edge_count,
                    "missing_edge_count": missing_edge_count,
                    "max_add_size_reached": k,
                    "candidates_tested": candidates_tested,
                    "pcg_supergraphs_found": pcg_supergraphs_found,
                    "runtime_seconds": round(elapsed, 6),
                }

            hmask = gmask | add_mask
            hg6 = graph6_from_mask(hmask, n, edge_to_index)

            batch_added_masks.append(add_mask)
            batch_graph6.append(hg6)

            if len(batch_graph6) >= candidate_batch_size:
                witness, tested_delta, pcg_delta = process_candidate_batch(
                    batch_added_masks=batch_added_masks,
                    batch_graph6=batch_graph6,
                    good_masks=good_masks,
                    oracle_nonpcg=oracle_nonpcg,
                    labelg_path=labelg_path,
                    edge_list=edge_list,
                )

                candidates_tested += tested_delta
                pcg_supergraphs_found += pcg_delta

                if witness is not None:
                    old_mask, new_mask, _ = witness
                    elapsed = time.perf_counter() - t0
                    return {
                        "target_index": target_index,
                        "graph6": g6,
                        "n": n,
                        "status_2and": "2-and-pcg",
                        "edge_count": edge_count,
                        "missing_edge_count": missing_edge_count,
                        "witness_added_mask_a": old_mask,
                        "witness_added_edges_a": edges_from_mask(old_mask, edge_list),
                        "witness_added_mask_b": new_mask,
                        "witness_added_edges_b": edges_from_mask(new_mask, edge_list),
                        "max_add_size_reached": k,
                        "candidates_tested": candidates_tested,
                        "pcg_supergraphs_found": pcg_supergraphs_found,
                        "runtime_seconds": round(elapsed, 6),
                    }

                batch_added_masks = []
                batch_graph6 = []

        if batch_graph6:
            witness, tested_delta, pcg_delta = process_candidate_batch(
                batch_added_masks=batch_added_masks,
                batch_graph6=batch_graph6,
                good_masks=good_masks,
                oracle_nonpcg=oracle_nonpcg,
                labelg_path=labelg_path,
                edge_list=edge_list,
            )

            candidates_tested += tested_delta
            pcg_supergraphs_found += pcg_delta

            if witness is not None:
                old_mask, new_mask, _ = witness
                elapsed = time.perf_counter() - t0
                return {
                    "target_index": target_index,
                    "graph6": g6,
                    "n": n,
                    "status_2and": "2-and-pcg",
                    "edge_count": edge_count,
                    "missing_edge_count": missing_edge_count,
                    "witness_added_mask_a": old_mask,
                    "witness_added_edges_a": edges_from_mask(old_mask, edge_list),
                    "witness_added_mask_b": new_mask,
                    "witness_added_edges_b": edges_from_mask(new_mask, edge_list),
                    "max_add_size_reached": k,
                    "candidates_tested": candidates_tested,
                    "pcg_supergraphs_found": pcg_supergraphs_found,
                    "runtime_seconds": round(elapsed, 6),
                }

        if debug_k:
            print(
                f"[worker-debug] graph6={g6} k={k} tested={candidates_tested} "
                f"pcg-supergraphs={pcg_supergraphs_found}",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.perf_counter() - t0

    if effective_max_k == missing_edge_count:
        return {
            "target_index": target_index,
            "graph6": g6,
            "n": n,
            "status_2and": "proven-non-2and",
            "edge_count": edge_count,
            "missing_edge_count": missing_edge_count,
            "max_add_size_reached": effective_max_k,
            "candidates_tested": candidates_tested,
            "pcg_supergraphs_found": pcg_supergraphs_found,
            "runtime_seconds": round(elapsed, 6),
        }

    return {
        "target_index": target_index,
        "graph6": g6,
        "n": n,
        "status_2and": "unresolved",
        "reason": "max-add-size",
        "edge_count": edge_count,
        "missing_edge_count": missing_edge_count,
        "max_add_size_reached": effective_max_k,
        "candidates_tested": candidates_tested,
        "pcg_supergraphs_found": pcg_supergraphs_found,
        "runtime_seconds": round(elapsed, 6),
    }


# ---------------------------------------------------------------------
# Multiprocessing.
# ---------------------------------------------------------------------

def init_worker(config: dict) -> None:
    """
    Initializer for forked workers.

    The large oracle set is inherited through fork as GLOBAL_ORACLE_NONPCG.
    We do not pass it as an initializer argument to avoid pickling it.
    """
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config


def worker_process_one(item: Tuple[int, dict]) -> dict:
    target_index, rec = item

    return test_target_2and_by_negative_oracle(
        target_index=target_index,
        rec=rec,
        n=GLOBAL_CONFIG["n"],
        edge_list=GLOBAL_CONFIG["edge_list"],
        edge_to_index=GLOBAL_CONFIG["edge_to_index"],
        all_edge_mask=GLOBAL_CONFIG["all_edge_mask"],
        oracle_nonpcg=GLOBAL_ORACLE_NONPCG,
        labelg_path=GLOBAL_CONFIG["labelg_path"],
        max_add_size=GLOBAL_CONFIG["max_add_size"],
        timeout_per_graph=GLOBAL_CONFIG["timeout_per_graph"],
        candidate_batch_size=GLOBAL_CONFIG["candidate_batch_size"],
        debug_k=GLOBAL_CONFIG["debug_k"],
    )


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel non-2-AND-PCG search using a non-PCG negative oracle."
    )

    p.add_argument("--n", type=int, default=10)

    p.add_argument(
        "--oracle-nonpcg",
        type=Path,
        default=None,
        help="Complete non-PCG JSONL file. Default: results/vert_n/non_pcg_n.jsonl",
    )
    p.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="Target non-PCG JSONL file. Default: results/vert_n/non_pcg_n.jsonl",
    )

    p.add_argument(
        "--proven-non2and-out",
        type=Path,
        default=None,
        help="Output JSONL for proven non-2-AND-PCGs.",
    )
    p.add_argument(
        "--unresolved-out",
        type=Path,
        default=None,
        help="Output JSONL for unresolved targets.",
    )
    p.add_argument(
        "--positive-out",
        type=Path,
        default=None,
        help="Optional output JSONL for found 2-AND-PCGs.",
    )
    p.add_argument(
        "--write-positive",
        action="store_true",
        help="Store found 2-AND-PCGs. By default, positives are only counted.",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary output path.",
    )

    p.add_argument(
        "--labelg",
        type=str,
        default=None,
        help="Path/name of nauty labelg.",
    )

    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel workers. On CloudLab ibm8335 try 120 first, then 160.",
    )

    p.add_argument(
        "--max-add-size",
        type=int,
        default=5,
        help=(
            "Maximum size of added-edge masks to search. "
            "Use -1 for exhaustive search."
        ),
    )
    p.add_argument(
        "--timeout-per-graph",
        type=float,
        default=60.0,
        help="Timeout per target graph in seconds. Use -1 for no timeout.",
    )
    p.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1000,
        help="Number of candidate supergraphs canonicalized per labelg call.",
    )
    p.add_argument(
        "--oracle-batch-size",
        type=int,
        default=100000,
        help="Number of oracle graph6 strings canonicalized per labelg call.",
    )
    p.add_argument(
        "--trust-oracle-canonical",
        action="store_true",
        help=(
            "Skip canonicalizing oracle graph6 strings. Faster, but only safe if "
            "oracle strings match labelg canonical output."
        ),
    )

    p.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every this many finished targets.",
    )
    p.add_argument(
        "--limit-targets",
        type=int,
        default=None,
        help="Optional limit for pilot runs.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Pool imap_unordered chunksize. Keep 1 when runtimes vary.",
    )
    p.add_argument(
        "--debug-k",
        action="store_true",
        help="Print per-target/per-k debug logs from workers. Very noisy.",
    )

    return p.parse_args()


def main() -> int:
    global GLOBAL_ORACLE_NONPCG

    args = parse_args()

    n = args.n
    edge_list, edge_to_index = build_edge_tables(n)
    all_edge_mask = (1 << len(edge_list)) - 1

    oracle_path = args.oracle_nonpcg or Path(f"results/vert_{n}/non_pcg_{n}.jsonl")
    targets_path = args.targets or Path(f"results/vert_{n}/non_pcg_{n}.jsonl")

    proven_out = args.proven_non2and_out or Path(
        f"results/vert_{n}/proven_non_2and_pcg_{n}.jsonl"
    )
    unresolved_out = args.unresolved_out or Path(
        f"results/vert_{n}/unresolved_2and_pcg_{n}.jsonl"
    )
    positive_out = args.positive_out or Path(
        f"results/vert_{n}/positive_2and_pcg_{n}.jsonl"
    )
    summary_path = args.summary or Path(
        f"results/vert_{n}/non2and_oracle_parallel_summary.txt"
    )

    for path in (oracle_path, targets_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    labelg_path = find_labelg(args.labelg)

    if args.max_add_size == -1:
        max_add_size: Optional[int] = None
    else:
        max_add_size = args.max_add_size

    if args.timeout_per_graph == -1:
        timeout_per_graph: Optional[float] = None
    else:
        timeout_per_graph = args.timeout_per_graph

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    if args.candidate_batch_size < 1:
        raise ValueError("--candidate-batch-size must be >= 1")

    if args.oracle_batch_size < 1:
        raise ValueError("--oracle-batch-size must be >= 1")

    start_all = time.perf_counter()

    print("=" * 72, file=sys.stderr)
    print("[config]", file=sys.stderr)
    print(f"n:                     {n}", file=sys.stderr)
    print(f"oracle_nonpcg:          {oracle_path}", file=sys.stderr)
    print(f"targets:                {targets_path}", file=sys.stderr)
    print(f"labelg:                 {labelg_path}", file=sys.stderr)
    print(f"jobs:                   {args.jobs}", file=sys.stderr)
    print(f"max_add_size:           {max_add_size if max_add_size is not None else 'exhaustive'}", file=sys.stderr)
    print(f"timeout_per_graph:      {timeout_per_graph if timeout_per_graph is not None else 'none'}", file=sys.stderr)
    print(f"candidate_batch_size:   {args.candidate_batch_size}", file=sys.stderr)
    print(f"oracle_batch_size:      {args.oracle_batch_size}", file=sys.stderr)
    print(f"trust_oracle_canonical: {args.trust_oracle_canonical}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    GLOBAL_ORACLE_NONPCG = load_oracle_nonpcg_set(
        oracle_path,
        n=n,
        edge_to_index=edge_to_index,
        labelg_path=labelg_path,
        canonical_batch_size=args.oracle_batch_size,
        trust_oracle_canonical=args.trust_oracle_canonical,
    )

    print(
        f"[info] canonical non-PCG oracle size: {len(GLOBAL_ORACLE_NONPCG):,}",
        file=sys.stderr,
    )

    proven_out.parent.mkdir(parents=True, exist_ok=True)
    unresolved_out.parent.mkdir(parents=True, exist_ok=True)
    positive_out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear output files.
    proven_out.write_text("", encoding="utf-8")
    unresolved_out.write_text("", encoding="utf-8")
    if args.write_positive:
        positive_out.write_text("", encoding="utf-8")

    config = {
        "n": n,
        "edge_list": edge_list,
        "edge_to_index": edge_to_index,
        "all_edge_mask": all_edge_mask,
        "labelg_path": labelg_path,
        "max_add_size": max_add_size,
        "timeout_per_graph": timeout_per_graph,
        "candidate_batch_size": args.candidate_batch_size,
        "debug_k": args.debug_k,
    }

    total_processed = 0
    positive_count = 0
    proven_non2and_count = 0
    unresolved_count = 0

    total_targets_known: Optional[int] = None
    if args.limit_targets is not None:
        total_targets_known = args.limit_targets

    print("[info] starting target processing...", file=sys.stderr, flush=True)

    f_pos = None

    try:
        with proven_out.open("a", encoding="utf-8") as f_proven, \
             unresolved_out.open("a", encoding="utf-8") as f_unresolved:

            if args.write_positive:
                f_pos = positive_out.open("a", encoding="utf-8")

            if args.jobs == 1:
                GLOBAL_CONFIG.update(config)

                iterator = map(
                    worker_process_one,
                    iter_target_records(targets_path, n=n, limit=args.limit_targets),
                )

                for result in iterator:
                    total_processed += 1

                    status = result["status_2and"]

                    if status == "2-and-pcg":
                        positive_count += 1
                        if f_pos is not None:
                            f_pos.write(json.dumps(result, separators=(",", ":")) + "\n")

                    elif status == "proven-non-2and":
                        proven_non2and_count += 1
                        f_proven.write(json.dumps(result, separators=(",", ":")) + "\n")
                        f_proven.flush()

                    elif status == "unresolved":
                        unresolved_count += 1
                        f_unresolved.write(json.dumps(result, separators=(",", ":")) + "\n")
                        f_unresolved.flush()

                    else:
                        raise RuntimeError(f"Unknown status: {status}")

                    if (
                        total_processed % max(1, args.progress_every) == 0
                        or (total_targets_known is not None and total_processed == total_targets_known)
                    ):
                        elapsed = time.perf_counter() - start_all
                        rate = total_processed / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[progress] processed={total_processed:,}, "
                            f"2AND={positive_count:,}, "
                            f"proven-non2AND={proven_non2and_count:,}, "
                            f"unresolved={unresolved_count:,}, "
                            f"elapsed={format_hms(elapsed)}, rate={rate:.2f}/s",
                            file=sys.stderr,
                            flush=True,
                        )

            else:
                if sys.platform.startswith("linux"):
                    ctx = mp.get_context("fork")
                else:
                    raise RuntimeError(
                        "This script is designed for Linux fork-based multiprocessing "
                        "so the oracle set is shared copy-on-write. Run on CloudLab/Linux."
                    )

                with ctx.Pool(
                    processes=args.jobs,
                    initializer=init_worker,
                    initargs=(config,),
                ) as pool:

                    iterator = pool.imap_unordered(
                        worker_process_one,
                        iter_target_records(targets_path, n=n, limit=args.limit_targets),
                        chunksize=max(1, args.chunksize),
                    )

                    for result in iterator:
                        total_processed += 1

                        status = result["status_2and"]

                        if status == "2-and-pcg":
                            positive_count += 1
                            if f_pos is not None:
                                f_pos.write(json.dumps(result, separators=(",", ":")) + "\n")

                        elif status == "proven-non-2and":
                            proven_non2and_count += 1
                            f_proven.write(json.dumps(result, separators=(",", ":")) + "\n")
                            f_proven.flush()

                        elif status == "unresolved":
                            unresolved_count += 1
                            f_unresolved.write(json.dumps(result, separators=(",", ":")) + "\n")
                            f_unresolved.flush()

                        else:
                            raise RuntimeError(f"Unknown status: {status}")

                        if (
                            total_processed % max(1, args.progress_every) == 0
                            or (
                                total_targets_known is not None
                                and total_processed == total_targets_known
                            )
                        ):
                            elapsed = time.perf_counter() - start_all
                            rate = total_processed / elapsed if elapsed > 0 else 0.0
                            print(
                                f"[progress] processed={total_processed:,}, "
                                f"2AND={positive_count:,}, "
                                f"proven-non2AND={proven_non2and_count:,}, "
                                f"unresolved={unresolved_count:,}, "
                                f"elapsed={format_hms(elapsed)}, rate={rate:.2f}/s",
                                file=sys.stderr,
                                flush=True,
                            )

    finally:
        if f_pos is not None:
            f_pos.close()

    elapsed_all = time.perf_counter() - start_all

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"n: {n}\n")
        f.write(f"oracle_nonpcg_file: {oracle_path}\n")
        f.write(f"targets_file: {targets_path}\n")
        f.write(f"proven_non2and_output: {proven_out}\n")
        f.write(f"unresolved_output: {unresolved_out}\n")

        if args.write_positive:
            f.write(f"positive_output: {positive_out}\n")

        f.write(f"total_processed: {total_processed}\n")
        f.write(f"found_2and_pcg: {positive_count}\n")
        f.write(f"proven_non2and_pcg: {proven_non2and_count}\n")
        f.write(f"unresolved: {unresolved_count}\n")
        f.write(f"oracle_nonpcg_size: {len(GLOBAL_ORACLE_NONPCG)}\n")
        f.write(f"jobs: {args.jobs}\n")
        f.write(f"max_add_size: {max_add_size if max_add_size is not None else 'exhaustive'}\n")
        f.write(f"timeout_per_graph: {timeout_per_graph if timeout_per_graph is not None else 'none'}\n")
        f.write(f"candidate_batch_size: {args.candidate_batch_size}\n")
        f.write(f"oracle_batch_size: {args.oracle_batch_size}\n")
        f.write(f"trust_oracle_canonical: {args.trust_oracle_canonical}\n")
        f.write(f"elapsed_seconds: {elapsed_all:.6f}\n")
        f.write(f"elapsed_hms: {format_hms(elapsed_all)}\n")

    print("=" * 72, file=sys.stderr)
    print("[done]", file=sys.stderr)
    print(f"processed:          {total_processed:,}", file=sys.stderr)
    print(f"2-AND-PCG:          {positive_count:,}", file=sys.stderr)
    print(f"proven non-2-AND:   {proven_non2and_count:,}", file=sys.stderr)
    print(f"unresolved:         {unresolved_count:,}", file=sys.stderr)
    print(f"elapsed:            {format_hms(elapsed_all)}", file=sys.stderr)
    print(f"summary:            {summary_path}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
