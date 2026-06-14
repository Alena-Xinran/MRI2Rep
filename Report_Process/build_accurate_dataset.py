#!/usr/bin/env python3
"""
build_accurate_dataset.py

Builds the most accurate possible dataset from all processed reports.

Tiered acceptance strategy (strictest first):
  Tier 1 — 3/3 consistent, valid vocab       → accept directly
  Tier 2 — 2/3, qty_only disagreement        → accept (type+pos confirmed by 2 rounds)
  Tier 3 — 2/3, diff==1 token                → accept (minor noise)
  Tier 4 — 2/3, diff==2 tokens               → accept cautiously
  Tier 5 — 2/3, diff>=3 tokens               → reject, add to rerun list
  Tier 6 — 1/3 agree                         → reject, add to rerun list (more rounds needed)

  All accepted cases: vocab validated, no multi-position strings allowed.

Output:
  accurate_summary.csv      — accepted cases (id, tier, sequence)
  rerun_needed.txt          — IDs requiring more LLM rounds
"""

import os
import json
import csv
import glob
import argparse

VALID_TYPES = {'APHE_WO', 'APHE_NoWO', 'RIM_ATYP', 'HEM', 'CYST'}
VALID_POS   = {'L_LAT', 'L_MED', 'R_ANT', 'R_POST', 'R_JUNCTION',
               'CAUDATE', 'L_DIFFUSE', 'R_DIFFUSE', 'DIFFUSE'}
VALID_QTY   = {'1', '2', 'GE3', 'Multiple'}

TYPE_PRIORITY = {'APHE_WO':1,'APHE_NoWO':2,'RIM_ATYP':3,'HEM':4,'CYST':5,'Unknown':99}
POS_PRIORITY  = {'CAUDATE':1,'L_LAT':2,'L_MED':3,'R_ANT':4,'R_POST':5,
                 'R_JUNCTION':6,'L_DIFFUSE':7,'R_DIFFUSE':8,'DIFFUSE':9,'Unknown':99}


def qty_normalize(q):
    q = str(q).strip()
    if 'multiple' in q.lower():
        return 'Multiple'
    try:
        n = int(q)
        if n <= 0:   return '1'
        if n == 1:   return '1'
        if n == 2:   return '2'
        return 'GE3'
    except ValueError:
        return 'Multiple'


def merge_tumors(tumor_list):
    grouped = {}
    for t in tumor_list:
        tp  = t.get('type', 'Unknown').strip()
        pos = t.get('position', 'Unknown').strip()
        qty = t.get('quantity', '1').strip()
        grouped.setdefault((tp, pos), []).append(qty)
    merged = []
    for (tp, pos), qtys in grouped.items():
        total = 0
        final_q = None
        for q in qtys:
            if 'multiple' in q.lower():
                final_q = 'Multiple'
                break
            try:
                total += int(q)
            except ValueError:
                final_q = 'Multiple'
                break
        if final_q is None:
            final_q = qty_normalize(total)
        merged.append({'type': tp, 'position': pos, 'quantity': final_q})
    merged.sort(key=lambda x: (TYPE_PRIORITY.get(x['type'], 99),
                                POS_PRIORITY.get(x['position'], 99)))
    return merged


def is_vocab_valid(tumors):
    """Check all tokens are in valid vocab and no multi-position strings."""
    for t in tumors:
        if ',' in t.get('position', ''):   # malformed: "L_MED, R_ANT"
            return False
        if t.get('type') not in VALID_TYPES:
            return False
        if t.get('position') not in VALID_POS:
            return False
    return True


def format_seq(organ, tumors):
    parts = [organ]
    for t in tumors:
        parts += [t['type'], t['position'], t['quantity']]
    return ', '.join(parts)


def classify_disagreement(details):
    """
    Returns (winner_count, diff_type, max_diff) where:
      diff_type: 'qty_only' | 'small' | 'large'
      max_diff: max token-set difference vs winner
    """
    variants = details.get('details', {})
    if not variants:
        return 0, 'unknown', 999

    winner_key = max(variants, key=variants.get)
    winner_count = variants[winner_key]
    winner_set = set(json.loads(winner_key))

    max_diff = 0
    diff_type = 'qty_only'

    for key in variants:
        if key == winner_key:
            continue
        dissenter_set = set(json.loads(key))
        added   = dissenter_set - winner_set
        removed = winner_set - dissenter_set
        n_diff  = len(added) + len(removed)
        max_diff = max(max_diff, n_diff)

        # Check if only quantity differs (same type|pos pairs)
        winner_pairs   = {tuple(r.split(' | ')[:2]) for r in winner_set}
        dissenter_pairs = {tuple(r.split(' | ')[:2]) for r in dissenter_set}
        if winner_pairs != dissenter_pairs:
            diff_type = 'large' if n_diff >= 3 else 'small'

    return winner_count, diff_type, max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_root', default='./logs')
    ap.add_argument('--out_csv',   default='./logs/accurate_summary.csv')
    ap.add_argument('--rerun_txt', default='./logs/rerun_needed.txt')
    ap.add_argument('--max_diff_accept', type=int, default=2,
                    help='Max token diff to accept 2/3 cases (default 2)')
    args = ap.parse_args()

    all_jsons = glob.glob(os.path.join(args.logs_root, 'batch_run_*', '*', 'final_result.json'))
    print(f'Found {len(all_jsons)} result files')

    accepted = []
    rerun    = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 'rejected_diff': 0,
                   'rejected_1of3': 0, 'rejected_vocab': 0}

    for path in sorted(all_jsons):
        with open(path) as f:
            d = json.load(f)

        case_id = d.get('id', 'Unknown')
        organ   = d.get('organ_status', 'Unknown')
        tumors  = merge_tumors(d.get('tumors', []))
        seq     = format_seq(organ, tumors)
        details = d.get('consistency_details', {})
        consistent = d.get('consistent', False)
        wc, diff_type, max_diff = classify_disagreement(details)

        # Vocab check (all tiers)
        if not is_vocab_valid(tumors):
            tier_counts['rejected_vocab'] += 1
            rerun.append(case_id)
            continue

        if consistent:
            # Tier 1: 3/3 agree
            accepted.append({'id': case_id, 'tier': 1, 'wc': wc, 'sequence': seq})
            tier_counts[1] += 1

        elif wc >= 2:
            if diff_type == 'qty_only':
                # Tier 2: only quantity differs, type+pos confirmed
                accepted.append({'id': case_id, 'tier': 2, 'wc': wc, 'sequence': seq})
                tier_counts[2] += 1
            elif max_diff == 1:
                # Tier 3: 1 token difference
                accepted.append({'id': case_id, 'tier': 3, 'wc': wc, 'sequence': seq})
                tier_counts[3] += 1
            elif max_diff <= args.max_diff_accept:
                # Tier 4: 2 tokens difference
                accepted.append({'id': case_id, 'tier': 4, 'wc': wc, 'sequence': seq})
                tier_counts[4] += 1
            else:
                # Too much disagreement → re-run with more rounds
                tier_counts['rejected_diff'] += 1
                rerun.append(case_id)
        else:
            # 1/3 agree → needs more rounds
            tier_counts['rejected_1of3'] += 1
            rerun.append(case_id)

    # Write accepted CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'tier', 'wc', 'sequence'])
        writer.writeheader()
        writer.writerows(sorted(accepted, key=lambda x: x['id']))

    # Write rerun list
    with open(args.rerun_txt, 'w') as f:
        f.write('\n'.join(sorted(rerun)))

    print(f'\n{"="*55}')
    print(f'ACCEPTED:')
    print(f'  Tier 1 (3/3 consistent):         {tier_counts[1]:>5}')
    print(f'  Tier 2 (2/3, qty_only diff):     {tier_counts[2]:>5}')
    print(f'  Tier 3 (2/3, diff=1 token):      {tier_counts[3]:>5}')
    print(f'  Tier 4 (2/3, diff=2 tokens):     {tier_counts[4]:>5}')
    print(f'  Total accepted:                  {len(accepted):>5}')
    print(f'\nREJECTED (need rerun):')
    print(f'  2/3 but diff≥3 tokens:           {tier_counts["rejected_diff"]:>5}')
    print(f'  1/3 agree (all 3 rounds differ): {tier_counts["rejected_1of3"]:>5}')
    print(f'  Invalid vocab:                   {tier_counts["rejected_vocab"]:>5}')
    print(f'  Total to rerun:                  {len(rerun):>5}')
    print(f'\nOutput: {args.out_csv}')
    print(f'Rerun list: {args.rerun_txt}')


if __name__ == '__main__':
    main()
