from pathlib import Path
import json
import os
import sys
from tqdm import tqdm


ROOT_DIR = Path("results")


def count_lines(path: Path) -> int:
    """Fast line counter for tqdm total."""
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def process_jsonl_file(jsonl_path: Path):
    txt_path = jsonl_path.with_suffix(".txt")
    tmp_path = jsonl_path.with_suffix(".txt.tmp")

    print(f"\nProcessing file: {jsonl_path}", flush=True)

    total_lines = count_lines(jsonl_path)

    try:
        with jsonl_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
            for line_no, line in enumerate(
                tqdm(
                    fin,
                    total=total_lines,
                    desc=jsonl_path.name,
                    unit="lines",
                    file=sys.stdout,
                ),
                start=1,
            ):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                if "graph6" not in obj:
                    raise KeyError(f"Missing 'graph6' at line {line_no} in {jsonl_path}")

                fout.write(obj["graph6"] + "\n")

        os.replace(tmp_path, txt_path)
        jsonl_path.unlink()

        print(f"Created: {txt_path}", flush=True)
        print(f"Deleted: {jsonl_path}", flush=True)

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()

        print(f"ERROR while processing {jsonl_path}", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        print("Original .jsonl file was NOT deleted.", file=sys.stderr)


def main():
    jsonl_files = sorted(ROOT_DIR.rglob("*.jsonl"))

    print(f"Found {len(jsonl_files)} JSONL files under {ROOT_DIR}/", flush=True)

    for jsonl_path in jsonl_files:
        process_jsonl_file(jsonl_path)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
