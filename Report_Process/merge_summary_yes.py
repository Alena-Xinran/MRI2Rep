#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
import argparse
from typing import List, Tuple, Set, Optional


def is_yes(x: str) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"yes", "y", "true", "1", "t"}


def detect_header(row: List[str]) -> bool:
    if not row:
        return False
    c0 = row[0].strip().lower()
    # 常见 header 情况
    return c0 in {"id", "case_id", "sample_id"}


def read_summary_csv(path: str) -> Tuple[Optional[List[str]], List[List[str]]]:
    """
    Returns: (header_or_none, rows)
    rows are raw columns, at least [id, consist, price, sequence] expected.
    """
    header = None
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all((c.strip() == "" for c in row)):
                continue
            # 如果第一行是 header
            if header is None and detect_header(row):
                header = row
                continue
            rows.append(row)
    return header, rows


def write_summary_csv(path: str, header: Optional[List[str]], rows: List[List[str]]) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
    os.replace(tmp, path)


def parse_batch_range_from_dirname(d: str) -> Tuple[int, int]:
    """
    batch_run_601_700 -> (601,700)
    If parsing fails, return (large, large) to push it to the end in sorting.
    """
    base = os.path.basename(d.rstrip("/"))
    m = re.search(r"batch_run_(\d+)_(\d+)", base)
    if not m:
        return (10**12, 10**12)
    return (int(m.group(1)), int(m.group(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="/home/xl754/MRI2Rep_PV_3F1/Report_Process/logs/",
                    help="root dir that contains batch_run_*/Summary.csv")
    ap.add_argument("--out_csv", default="/home/xl754/MRI2Rep_PV_3F1/Report_Process/logs/merged_summary_yes.csv",
                    help="merged output csv (will be updated in-place)")
    ap.add_argument("--limit_batches", type=int, default=0,
                    help="if >0, only merge the first N batch_run dirs after sorting by range")
    args = ap.parse_args()

    logs_root = args.logs_root
    out_csv = args.out_csv

    # 1) 读取已有总表（如果存在），建立已存在 id 集合
    out_header = None
    out_rows: List[List[str]] = []
    existing_ids: Set[str] = set()

    if os.path.exists(out_csv):
        out_header, out_rows = read_summary_csv(out_csv)
        for r in out_rows:
            if not r:
                continue
            existing_ids.add(r[0].strip())
    else:
        # 如果你希望总表带 header，可以在这里设置；否则保持 None（无表头）
        out_header = None
        out_rows = []

    # 2) 扫描所有 batch_run_*/Summary.csv（自动忽略不存在的）
    batch_dirs = glob.glob(os.path.join(logs_root, "batch_run_*"))
    batch_dirs.sort(key=parse_batch_range_from_dirname)

    if args.limit_batches and args.limit_batches > 0:
        batch_dirs = batch_dirs[:args.limit_batches]

    summary_paths = []
    for d in batch_dirs:
        p = os.path.join(d, "Summary.csv")
        if os.path.exists(p):
            # 避免把总表自己当成输入
            if os.path.abspath(p) != os.path.abspath(out_csv):
                summary_paths.append(p)

    # 3) 合并：只取 consist=Yes，且 id 不在 existing_ids 中
    newly_added = 0
    bad_rows = 0
    seen_this_run: Set[str] = set()

    for p in summary_paths:
        _, rows = read_summary_csv(p)
        for r in rows:
            # 期望至少 2 列：[id, consist, ...]
            if len(r) < 2:
                bad_rows += 1
                continue
            rid = r[0].strip()
            consist = r[1]

            if rid == "":
                bad_rows += 1
                continue
            if not is_yes(consist):
                continue

            if rid in existing_ids or rid in seen_this_run:
                continue

            out_rows.append(r)
            existing_ids.add(rid)
            seen_this_run.add(rid)
            newly_added += 1

    # 4) 写回总表（原子替换，避免中途写坏文件）
    write_summary_csv(out_csv, out_header, out_rows)

    print(f"[OK] logs_root: {logs_root}")
    print(f"[OK] found batch summaries: {len(summary_paths)}")
    print(f"[OK] newly appended (consist=Yes & new id): {newly_added}")
    print(f"[OK] total rows in merged Summary.csv: {len(out_rows)}")
    if bad_rows > 0:
        print(f"[WARN] skipped bad/short rows: {bad_rows}")


if __name__ == "__main__":
    main()
