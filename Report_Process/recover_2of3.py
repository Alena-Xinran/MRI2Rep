#!/usr/bin/env python3
"""
recover_2of3.py

Two jobs:
  1. Recover cases where 2/3 rounds agreed (winner_count >= MIN_WINNER).
     These already have correct data in final_result.json — no API call needed.
     Writes a new merged_summary_recovered.csv alongside the existing one.

  2. Print a list of case IDs where only 1/3 rounds agreed → need re-run.
     Saves them to rerun_ids.txt for use by rerun_failed.py.

Usage:
    python recover_2of3.py
    python recover_2of3.py --min_winner 2 --logs_root ./logs --out_csv ./logs/merged_summary_recovered.csv
"""

import os
import json
import csv
import glob
import argparse
from collections import Counter

# ── Same sorting logic as post.py ────────────────────────────────────────────
TYPE_PRIORITY = {
    "APHE_WO": 1, "APHE_NoWO": 2, "RIM_ATYP": 3,
    "HEM": 4, "CYST": 5, "FAT": 6, "OTHER": 99, "Unknown": 100
}
POS_PRIORITY = {
    "CAUDATE": 1, "L_LAT": 2, "L_MED": 3, "R_ANT": 4, "R_POST": 5,
    "R_JUNCTION": 6, "L_DIFFUSE": 7, "R_DIFFUSE": 8, "DIFFUSE": 9, "Unknown": 99
}

def calculate_merged_quantity(quantities):
    total = 0
    for q in quantities:
        q_str = str(q).strip()
        if "multiple" in q_str.lower():
            return "Multiple"
        try:
            total += int(q_str)
        except ValueError:
            return "Multiple"
    return str(total)

def merge_tumors(tumor_list):
    if not tumor_list:
        return []
    grouped = {}
    for t in tumor_list:
        key = (t.get("type","Unknown").strip(), t.get("position","Unknown").strip())
        grouped.setdefault(key, []).append(t.get("quantity","1").strip())
    merged = [{"type": k[0], "position": k[1],
               "quantity": calculate_merged_quantity(v)}
              for k, v in grouped.items()]
    merged.sort(key=lambda x: (TYPE_PRIORITY.get(x["type"], 100),
                                POS_PRIORITY.get(x["position"], 100)))
    return merged

def format_sequence(organ_status, tumors):
    parts = [organ_status]
    for t in tumors:
        parts += [t["type"], t["position"], t["quantity"]]
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="./logs")
    ap.add_argument("--out_csv",   default="./logs/merged_summary_recovered.csv")
    ap.add_argument("--rerun_txt", default="./logs/rerun_ids.txt")
    ap.add_argument("--min_winner", type=int, default=2,
                    help="Accept as consistent if winner_count >= this (default 2)")
    args = ap.parse_args()

    all_jsons = glob.glob(os.path.join(args.logs_root, "batch_run_*", "*", "final_result.json"))
    print(f"Found {len(all_jsons)} final_result.json files")

    rows_yes_original = []   # originally consistent=True
    rows_recovered    = []   # was False, but winner_count >= min_winner
    rerun_ids         = []   # winner_count < min_winner → need re-run

    for path in sorted(all_jsons):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}")
            continue

        case_id   = data.get("id", "Unknown")
        consistent = data.get("consistent", False)
        organ     = data.get("organ_status", "Unknown")
        tumors    = merge_tumors(data.get("tumors", []))
        sequence  = format_sequence(organ, tumors)
        cost      = data.get("_meta", {}).get("total_cost_usd", 0.0)
        details   = data.get("consistency_details", {})
        wc        = details.get("winner_count", 0)
        rounds    = details.get("total_rounds", 3)

        row = {"id": case_id, "consistency": "Yes", "cost": f"{cost:.6f}",
               "sequence": sequence, "winner_count": wc, "total_rounds": rounds}

        if consistent:
            rows_yes_original.append(row)
        elif wc >= args.min_winner:
            rows_recovered.append(row)
        else:
            rerun_ids.append(case_id)

    # ── Write recovered CSV ───────────────────────────────────────────────────
    all_rows = rows_yes_original + rows_recovered
    all_rows.sort(key=lambda x: x["id"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","consistency","cost","sequence","winner_count","total_rounds"])
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Write rerun list ──────────────────────────────────────────────────────
    with open(args.rerun_txt, "w") as f:
        f.write("\n".join(rerun_ids))

    print(f"\n{'='*50}")
    print(f"Originally consistent (3/3):  {len(rows_yes_original)}")
    print(f"Recovered ({args.min_winner}/3 agree):       {len(rows_recovered)}")
    print(f"Need re-run (<{args.min_winner}/3 agree):    {len(rerun_ids)}")
    print(f"Total usable now:             {len(all_rows)}")
    print(f"\nSaved recovered CSV → {args.out_csv}")
    print(f"Saved re-run list   → {args.rerun_txt}")


if __name__ == "__main__":
    main()
