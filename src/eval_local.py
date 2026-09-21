"""
eval_local.py
Local scorer for Vaani Noise Event Detection (IndoML 2026, Track 1).

Reproduces the Codabench leaderboard metric locally so we can sanity-check
predictions before spending a submission:

    Combined = Event-based F1 (onset/offset within ±20% tolerance)
             + Segment-level Dice (frame-level overlap)
    (max possible score = 2.0)

Usage:
    python src/eval_local.py \
        --predictions submissions/predictions.jsonl \
        --ground-truth data/processed/val_split.jsonl \
        --tolerance 0.2 \
        --frame-size 0.01

Input formats
-------------
Ground truth (val_split.jsonl / the Codabench "Validation Dataset"), one
JSON object per line:
    {"clip_id": "...", "duration": 3.39, "events": [{"onset": 0.0, "offset": 3.38, "category": "..."}]}

Predictions (submissions/predictions.jsonl), one JSON object per line:
    {"clip_id": "...", "events": [{"onset": 1.24, "offset": 3.81}]}

Notes / assumptions (flag if Codabench's exact definitions differ)
--------------------------------------------------------------------
- Event-based matching: a predicted event matches a ground-truth event if
  they're on the same clip and BOTH:
    |pred_onset  - gt_onset|  <= tolerance * gt_event_duration
    |pred_offset - gt_offset| <= tolerance * gt_event_duration
  Matching is done greedily, one-to-one, per clip (each GT event can be
  matched by at most one prediction and vice versa), closest-onset-first.
  Precision/Recall/F1 are then computed over the whole dataset (micro-avg).
- Segment-level Dice: each clip's timeline is rasterized into fixed-size
  frames (default 10ms). Dice = 2*|pred ∩ gt| / (|pred| + |gt|) in frame
  counts, per clip, then averaged over clips (macro-avg). A clip with no
  GT events and no predicted events scores Dice = 1.0 for that clip.
- Clips present in ground truth but missing from predictions are scored as
  "no events predicted" (0 recall contribution, Dice computed against an
  empty prediction) rather than skipped.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Local evaluator for Vaani Noise Event Detection")
    p.add_argument("--predictions", type=str, required=True,
                    help="Path to predictions.jsonl")
    p.add_argument("--ground-truth", type=str, required=True,
                    help="Path to ground-truth JSONL (e.g. val_split.jsonl or the Codabench Validation Dataset labels)")
    p.add_argument("--tolerance", type=float, default=0.2,
                    help="Onset/offset tolerance as a fraction of GT event duration (default 0.2 = ±20%%)")
    p.add_argument("--frame-size", type=float, default=0.01,
                    help="Frame size in seconds for segment-level Dice rasterization (default 10ms)")
    p.add_argument("--per-clip", action="store_true",
                    help="Print per-clip Event-F1 / Dice / Combined instead of just the aggregate")
    return p.parse_args()


def load_jsonl(path: str) -> Dict[str, dict]:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["clip_id"]] = rec
    return records


# --------------------------------------------------------------------- #
# Event-based F1 (±tolerance)
# --------------------------------------------------------------------- #

def match_events(gt_events: List[dict], pred_events: List[dict], tolerance: float) -> Tuple[int, int, int]:
    """
    Greedy one-to-one matching within a single clip.
    Returns (true_positives, false_positives, false_negatives) for this clip.
    """
    gt_used = [False] * len(gt_events)
    pred_used = [False] * len(pred_events)

    # Build all valid (pred_idx, gt_idx) candidate pairs, sorted by onset gap
    # so the closest matches are claimed first.
    candidates = []
    for pi, pe in enumerate(pred_events):
        p_on, p_off = pe["onset"], pe["offset"]
        for gi, ge in enumerate(gt_events):
            g_on, g_off = ge["onset"], ge["offset"]
            gt_dur = max(g_off - g_on, 1e-6)
            allowed = tolerance * gt_dur
            if abs(p_on - g_on) <= allowed and abs(p_off - g_off) <= allowed:
                gap = abs(p_on - g_on) + abs(p_off - g_off)
                candidates.append((gap, pi, gi))

    candidates.sort(key=lambda x: x[0])

    tp = 0
    for _, pi, gi in candidates:
        if not pred_used[pi] and not gt_used[gi]:
            pred_used[pi] = True
            gt_used[gi] = True
            tp += 1

    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    return tp, fp, fn


# --------------------------------------------------------------------- #
# Segment-level Dice
# --------------------------------------------------------------------- #

def events_to_frame_mask(events: List[dict], duration: float, frame_size: float) -> set:
    """Return the set of frame indices covered by any event."""
    n_frames = max(1, int(round(duration / frame_size)))
    covered = set()
    for ev in events:
        start_f = max(0, int(ev["onset"] / frame_size))
        end_f = min(n_frames, int(round(ev["offset"] / frame_size)))
        for f in range(start_f, end_f):
            covered.add(f)
    return covered


def clip_dice(gt_events: List[dict], pred_events: List[dict], duration: float, frame_size: float) -> float:
    gt_frames = events_to_frame_mask(gt_events, duration, frame_size)
    pred_frames = events_to_frame_mask(pred_events, duration, frame_size)

    if not gt_frames and not pred_frames:
        return 1.0
    intersection = len(gt_frames & pred_frames)
    denom = len(gt_frames) + len(pred_frames)
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    args = parse_args()

    gt = load_jsonl(args.ground_truth)
    preds = load_jsonl(args.predictions)

    total_tp = total_fp = total_fn = 0
    dice_scores = []
    per_clip_rows = []

    missing_in_preds = 0

    for clip_id, gt_rec in gt.items():
        gt_events = gt_rec.get("events", [])
        pred_rec = preds.get(clip_id)
        if pred_rec is None:
            missing_in_preds += 1
            pred_events = []
        else:
            pred_events = pred_rec.get("events", [])

        duration = gt_rec.get("duration")
        if duration is None:
            # Fall back to the furthest event offset if duration wasn't stored
            all_offsets = [e["offset"] for e in gt_events + pred_events] or [0.0]
            duration = max(all_offsets)

        tp, fp, fn = match_events(gt_events, pred_events, args.tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        dice = clip_dice(gt_events, pred_events, duration, args.frame_size)
        dice_scores.append(dice)

        if args.per_clip:
            precision = tp / (tp + fp) if (tp + fp) else 1.0
            recall = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            per_clip_rows.append((clip_id, f1, dice, f1 + dice))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    event_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    combined = event_f1 + mean_dice

    if args.per_clip:
        print(f"{'clip_id':<20} {'Event-F1':>10} {'Dice':>10} {'Combined':>10}")
        for clip_id, f1, dice, comb in per_clip_rows:
            print(f"{clip_id:<20} {f1:>10.4f} {dice:>10.4f} {comb:>10.4f}")
        print()

    print("=" * 50)
    print(f"Clips evaluated:      {len(gt)}")
    if missing_in_preds:
        print(f"Clips missing from predictions (scored as empty): {missing_in_preds}")
    print(f"TP / FP / FN:         {total_tp} / {total_fp} / {total_fn}")
    print(f"Event Precision:      {precision:.4f}")
    print(f"Event Recall:         {recall:.4f}")
    print(f"Event-based F1:       {event_f1:.4f}")
    print(f"Segment-level Dice:   {mean_dice:.4f}")
    print(f"Combined (max 2.0):   {combined:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
