from pathlib import Path
import concurrent.futures as cf
import json
import subprocess
import sys
import time

from tqdm import tqdm


def iter_connected_graph6_from_geng(n: int, geng_path: str = "nauty-geng"):
    """
    Yield connected unlabeled graph6 strings on n vertices using nauty geng.
    """
    cmd = [geng_path, "-q", "-c", str(n)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            g6 = line.strip()
            if g6 and not g6.startswith(">"):
                yield g6
    finally:
        if proc.stdout:
            proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(
                f"geng failed for n={n} with exit code {ret}.\nStderr:\n{stderr}"
            )


def analyze_worker(args):
    """
    Worker must import inside the process to keep Z3 isolated per process.
    """
    g6, verbose = args
    from pcg_library_star_cat_gen import analyze_one_graph6
    return analyze_one_graph6(g6, verbose=verbose)


def format_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def classify_graphs_on_n_vertices(
    n: int,
    geng_path: str = "nauty-geng",
    verbose: bool = False,
    max_workers: int | None = None,
    queue_multiplier: int = 2,
):
    """
    Enumerate all connected unlabeled graphs on n vertices, analyze each one with
    pcg_library_star_cat_gen.analyze_one_graph6(), and store results into:

        results/vert_{n}/pcg_star_{n}.jsonl
        results/vert_{n}/pcg_cat_{n}.jsonl
        results/vert_{n}/pcg_gen_{n}.jsonl
        results/vert_{n}/non_pcg_{n}.jsonl
        results/vert_{n}/summary.txt
    """
    start_time = time.perf_counter()

    if max_workers is None:
        import os
        max_workers = max(1, os.cpu_count() or 1)

    queue_limit = max(1, max_workers * max(1, queue_multiplier))

    outdir = Path("results") / f"vert_{n}"
    outdir.mkdir(parents=True, exist_ok=True)

    star_path = outdir / f"pcg_star_{n}.jsonl"
    cat_path = outdir / f"pcg_cat_{n}.jsonl"
    gen_path = outdir / f"pcg_gen_{n}.jsonl"
    non_path = outdir / f"non_pcg_{n}.jsonl"
    summary_path = outdir / "summary.txt"

    total = 0
    counts = {
        "sat-star": 0,
        "sat-caterpillar": 0,
        "sat-general": 0,
        "unsat": 0,
    }

    def handle_result(rec, f_star, f_cat, f_gen, f_non, pbar):
        nonlocal total
        total += 1

        status = rec["status"]
        line = json.dumps(rec, ensure_ascii=False)

        if status == "sat-star":
            f_star.write(line + "\n")
            counts["sat-star"] += 1
        elif status == "sat-caterpillar":
            f_cat.write(line + "\n")
            counts["sat-caterpillar"] += 1
        elif status == "sat-general":
            f_gen.write(line + "\n")
            counts["sat-general"] += 1
        elif status == "unsat":
            f_non.write(line + "\n")
            counts["unsat"] += 1
        else:
            raise ValueError(f"Unexpected status: {status}")

        pbar.update(1)

        if verbose:
            print(
                f"[classify_graphs_on_n_vertices] "
                f"#{total} graph6={rec.get('graph6', '<missing>')} status={status}"
            )

    with (
        star_path.open("w", encoding="utf-8") as f_star,
        cat_path.open("w", encoding="utf-8") as f_cat,
        gen_path.open("w", encoding="utf-8") as f_gen,
        non_path.open("w", encoding="utf-8") as f_non,
        cf.ProcessPoolExecutor(max_workers=max_workers) as ex,
    ):
        pending = {}
        g6_iter = iter_connected_graph6_from_geng(n, geng_path=geng_path)

        pbar = tqdm(desc=f"n={n}", unit="graph")

        try:
            while len(pending) < queue_limit:
                g6 = next(g6_iter)
                fut = ex.submit(analyze_worker, (g6, verbose))
                pending[fut] = g6
        except StopIteration:
            pass

        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)

            for fut in done:
                g6 = pending.pop(fut)
                try:
                    rec = fut.result()
                except Exception as e:
                    pbar.close()
                    raise RuntimeError(f"Worker failed on graph6={g6}") from e

                handle_result(rec, f_star, f_cat, f_gen, f_non, pbar)

            try:
                while len(pending) < queue_limit:
                    g6 = next(g6_iter)
                    fut = ex.submit(analyze_worker, (g6, verbose))
                    pending[fut] = g6
            except StopIteration:
                pass

        pbar.close()

    elapsed_seconds = time.perf_counter() - start_time

    summary = {
        "n": n,
        "output_dir": str(outdir),
        "pcg_star_file": str(star_path),
        "pcg_cat_file": str(cat_path),
        "pcg_gen_file": str(gen_path),
        "non_pcg_file": str(non_path),
        "summary_file": str(summary_path),
        "total": total,
        "sat-star": counts["sat-star"],
        "sat-caterpillar": counts["sat-caterpillar"],
        "sat-general": counts["sat-general"],
        "unsat": counts["unsat"],
        "elapsed_seconds": round(elapsed_seconds, 6),
        "elapsed_hms": format_elapsed(elapsed_seconds),
        "max_workers": max_workers,
        "queue_limit": queue_limit,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python classify_pcg.py <n> [geng_path] [max_workers] [verbose]\n"
            "Examples:\n"
            "  python classify_pcg.py 8\n"
            "  python classify_pcg.py 8 geng 16\n"
            "  python classify_pcg.py 8 nauty-geng 8 true"
        )
        sys.exit(1)

    n = int(sys.argv[1])
    geng_path = sys.argv[2] if len(sys.argv) >= 3 else "nauty-geng"
    max_workers = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    if len(sys.argv) >= 5:
        verbose = sys.argv[4].strip().lower() in {"1", "true", "yes", "y"}
    else:
        verbose = False

    summary = classify_graphs_on_n_vertices(
        n=n,
        geng_path=geng_path,
        verbose=verbose,
        max_workers=max_workers,
    )
    print(json.dumps(summary, indent=2))
