#!/usr/bin/env python3
"""
rerun_failed.py

Re-runs the LLM pipeline for cases that had < 2/3 agreement.
Backs up the original final_result.json, then runs 5 rounds (instead of 3)
to give ambiguous cases a better chance of reaching majority consensus.

Usage:
    python rerun_failed.py --rerun_txt ./logs/rerun_ids.txt
    python rerun_failed.py --rerun_txt ./logs/rerun_ids.txt --rounds 5 --dry_run
"""

import os
import sys
import json
import shutil
import argparse
import glob
import re
from collections import Counter

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config
import pipeline
from utils.logger import AnalysisLogger
import utils.io as utils_io

TYPE_PRIORITY = {
    "APHE_WO": 1, "APHE_NoWO": 2, "RIM_ATYP": 3,
    "HEM": 4, "CYST": 5, "FAT": 6, "OTHER": 99, "Unknown": 100
}
POS_PRIORITY = {
    "CAUDATE": 1, "L_LAT": 2, "L_MED": 3, "R_ANT": 4, "R_POST": 5,
    "R_JUNCTION": 6, "L_DIFFUSE": 7, "R_DIFFUSE": 8, "DIFFUSE": 9, "Unknown": 99
}

def find_case_dir(logs_root, case_id):
    """Find the batch directory that contains this case."""
    matches = glob.glob(os.path.join(logs_root, "batch_run_*", case_id))
    return matches[0] if matches else None

def find_report_file(case_id):
    """Search all split dirs for the report file."""
    pattern = os.path.join("./reports/all_splits", "*", f"{case_id}*")
    matches = glob.glob(pattern)
    if not matches:
        # try without subfolder
        matches = glob.glob(os.path.join("./reports", f"{case_id}*"))
    return matches[0] if matches else None

def rerun_case(case_id, logs_root, rounds, dry_run=False):
    """Back up old result and re-run pipeline with more rounds."""
    case_dir = find_case_dir(logs_root, case_id)
    if not case_dir:
        print(f"  [SKIP] Cannot find case directory for {case_id}")
        return None

    result_path = os.path.join(case_dir, "final_result.json")
    report_file = find_report_file(case_id)

    if not report_file:
        print(f"  [SKIP] Cannot find report file for {case_id}")
        return None

    if dry_run:
        print(f"  [DRY] Would rerun {case_id} ({rounds} rounds) from {report_file}")
        return "dry_run"

    # Backup original result
    backup_path = result_path + ".backup"
    if os.path.exists(result_path) and not os.path.exists(backup_path):
        shutil.copy2(result_path, backup_path)

    print(f"\n🔄 Re-running {case_id} with {rounds} rounds...")

    try:
        report_text = utils_io.read_file_content(report_file)
    except Exception as e:
        print(f"  [ERROR] Cannot read report: {e}")
        return None

    # Temporarily override CONSISTENCY_ROUNDS and RUN_ID to match case's actual batch dir
    original_rounds = config.CONSISTENCY_ROUNDS
    original_run_id = config.RUN_ID
    config.CONSISTENCY_ROUNDS = rounds
    config.RUN_ID = os.path.basename(os.path.dirname(case_dir))

    logger = AnalysisLogger(case_id)
    logger.setup()  # ensure the case log directory exists

    try:
        # Step 3 only (organ status from backup is still good)
        with open(backup_path if os.path.exists(backup_path) else result_path) as f:
            old_data = json.load(f)
        organ_status = old_data.get("organ_status", "Unknown")

        raw_tumors, consistent, stats = pipeline.run_step_3_tumor_extraction(report_text, logger)
        structured = pipeline.parse_tumor_records(raw_tumors)

        final_data = {
            "id": case_id,
            "organ_status": organ_status,
            "tumors": structured,
            "consistent": consistent,
            "consistency_details": stats,
            "_meta": old_data.get("_meta", {})
        }

        with open(result_path, "w") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        wc = stats.get("winner_count", 0)
        print(f"  Result: winner_count={wc}/{rounds}, consistent={consistent}")
        return consistent

    except Exception as e:
        print(f"  [ERROR] {e}")
        # Restore backup on error
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, result_path)
        return None
    finally:
        config.CONSISTENCY_ROUNDS = original_rounds
        config.RUN_ID = original_run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun_txt", default="./logs/rerun_ids.txt")
    ap.add_argument("--logs_root", default="./logs")
    ap.add_argument("--rounds",    type=int, default=5,
                    help="Number of LLM rounds for re-run (default 5)")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Print what would be done without calling LLM")
    ap.add_argument("--limit",     type=int, default=0,
                    help="Only rerun first N cases (for testing)")
    args = ap.parse_args()

    if not os.path.exists(args.rerun_txt):
        print(f"[ERROR] rerun_ids.txt not found: {args.rerun_txt}")
        print("Run recover_2of3.py first to generate it.")
        sys.exit(1)

    with open(args.rerun_txt) as f:
        ids = [line.strip() for line in f if line.strip()]

    if args.limit > 0:
        ids = ids[:args.limit]

    print(f"Re-running {len(ids)} cases with {args.rounds} rounds each")
    if args.dry_run:
        print("[DRY RUN MODE — no API calls]")

    success = fail = skip = 0
    for i, case_id in enumerate(ids, 1):
        print(f"\n[{i}/{len(ids)}] {case_id}")
        result = rerun_case(case_id, args.logs_root, args.rounds, args.dry_run)
        if result is None:
            skip += 1
        elif result == "dry_run":
            pass
        elif result:
            success += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"Re-run complete:")
    print(f"  Now consistent: {success}")
    print(f"  Still failed:   {fail}")
    print(f"  Skipped:        {skip}")
    print(f"\nRun recover_2of3.py again to regenerate the merged CSV.")


if __name__ == "__main__":
    main()
