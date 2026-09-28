#!/usr/bin/env python3
"""Compare complete raw-logit hashes emitted by gufo bench --logit-eval.

Usage: python tools/bench/logit-eval.py LEFT.bin.sha256 RIGHT.bin.sha256
Exit 0 means equal rows, 1 means differing rows, and 2 means invalid input.
This checks the recorded teacher-forced positions, not rejection rollback.
"""
import argparse
from pathlib import Path
import re
import sys


def read_dump(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty dump")
    header = lines[0].split()
    if len(header) != 4 or header[:2] != ["GFLE-SHA256", "1"]:
        raise ValueError(f"{path}: unsupported header")
    vocab, count = map(int, header[2:])
    if vocab <= 0 or count <= 0 or len(lines) != count + 1:
        raise ValueError(f"{path}: invalid vocabulary or incomplete row count")
    rows = []
    for position, line in enumerate(lines[1:], 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}: malformed row {position}")
        row, target = map(int, fields[:2])
        if row != position or not 0 <= target < vocab or not re.fullmatch(r"[0-9a-f]{64}", fields[2]):
            raise ValueError(f"{path}: invalid row {position}")
        rows.append((target, fields[2]))
    return vocab, rows


def compare(left, right):
    vocab, rows = read_dump(left)
    other_vocab, other = read_dump(right)
    if vocab != other_vocab or [row[0] for row in rows] != [row[0] for row in other]:
        raise ValueError("dumps have different vocabularies, positions, or target tokens")
    return [i for i, (a, b) in enumerate(zip(rows, other), 1) if a[1] != b[1]], len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    try:
        different, count = compare(args.left, args.right)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if different:
        print(f"DIFFER: {len(different)}/{count} raw-logit rows; first positions: {different[:8]}")
        return 1
    print(f"MATCH: all {count} complete raw-logit rows have identical SHA-256 hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
