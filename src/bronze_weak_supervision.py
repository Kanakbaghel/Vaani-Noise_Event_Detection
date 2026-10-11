"""
bronze_weak_supervision.py
Bronze-tier weak supervision experiment for Vaani Noise Event Detection (Track 1).

Novelty angle: the ~30h Bronze tier has NO onset/offset timestamps -- only inline
noise tags in the transcript, e.g. "<horn> </horn>", telling us a clip DOES or
DOESN'T contain a noise event, but not where. That's a weak, clip-level label.

This script turns that into usable training data via self-training:

  1. Run our best trained CRNN (Gold+Silver) over every Bronze clip to get its
     own frame-level predictions (pseudo onset/offset events), using the exact
     same decoding logic (logits_to_events) as infer.py -- so pseudo-labels are
     produced the same way our real submissions are.
  2. Keep a Bronze clip only when the model's own prediction AGREES with the
     weak tag-presence label:
       - tags say "noise present" AND model confidently predicts >=1 event
         -> accept, pseudo-label = model's predicted events
       - tags say "no noise" AND model confidently predicts no events
         -> accept as a confident NEGATIVE example (events = [])
       - disagreement (tags say noise but model found none, or vice versa)
         -> discard; too risky to trust either side alone
  3. Write the accepted, pseudo-labeled Bronze clips to a new JSONL in the same
     schema as Gold/Silver (clip_id, tier, events), ready to feed straight into
     VaaniNoiseDataset / train.py via --tier-filter.

This is a filtering self-training approach, not a new architecture -- the
novelty is using the previously-untouched Bronze tier at all, and only trusting
it where the model and the weak label corroborate each other.

Usage:
    python src/bronze_weak_supervision.py \
        --unified-jsonl data/processed/unified.jsonl \
        --model models/baseline_crnn_9426_gold.pt \
        --out data/processed/bronze_pseudolabeled.jsonl \
        --confidence 0.6 \
        --max-clips 5000

Then, to actually use it:
    cat data/processed/train_split.jsonl data/processed/bronze_pseudolabeled.jsonl \
        > data/processed/train_split_with_bronze.jsonl
    python src/train.py --train-jsonl data/processed/train_split_with_bronze.jsonl \
        --tier-filter gold silver bronze_pseudo \
        --save-path models/baseline_crnn_with_bronze.pt
    python src/eval_local.py --predictions <preds from the new model> \
        --ground-truth data/processed/val_split.jsonl
Compare the Combined score against the current 0.8418 champion (Gold+Silver
only) to see whether the Bronze augmentation actually helps.

NOTE: this script needs HF_TOKEN-backed audio access (same as the rest of the
pipeline) and a trained checkpoint -- neither is available in this drafting
environment, so the logic below is built and smoke-tested against fabricated
inputs, not run against the real Bronze tier. Run it for real on a machine with
the .env token and models/ checkpoint before trusting the output counts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import LogMelExtractor
from src.model import CRNNNoiseDetector
from src.infer import logits_to_events, load_checkpoint  # reuse the team's own decoding logic
from src.audio_loader import AudioLoader


def parse_args():
    p = argparse.ArgumentParser(description="Bronze-tier weak supervision via self-training + tag agreement")
    p.add_argument("--unified-jsonl", type=str, default="data/processed/unified.jsonl",
                    help="Output of data_prep.py, containing all tiers")
    p.add_argument("--model", type=str, required=True,
                    help="Path to the best trained CRNN checkpoint (Gold+Silver)")
    p.add_argument("--out", type=str, default="data/processed/bronze_pseudolabeled.jsonl",
                    help="Where to write accepted, pseudo-labeled Bronze clips")
    p.add_argument("--cache-dir", type=str, default="data/cache/audio_clips")
    p.add_argument("--threshold", type=float, default=0.35,
                    help="Event probability threshold (match the team's tuned infer.py config)")
    p.add_argument("--median-filter", type=int, default=7)
    p.add_argument("--min-duration", type=float, default=0.15)
    p.add_argument("--merge-gap", type=float, default=0.10)
    p.add_argument("--confidence", type=float, default=0.6,
                    help="Minimum mean probability over predicted-positive frames to trust a positive clip")
    p.add_argument("--max-clips", type=int, default=None,
                    help="Cap how many Bronze clips to scan (for a quicker first pass)")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def load_bronze_records(unified_jsonl, max_clips=None):
    records = []
    with open(unified_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("tier") == "bronze":
                records.append(rec)
                if max_clips and len(records) >= max_clips:
                    break
    return records


def mean_confidence_on_mask(probs: np.ndarray, events, hop_length, sr):
    """Average predicted probability over the frames covered by the decoded events."""
    if not events:
        return 0.0
    n_frames = len(probs)
    covered = np.zeros(n_frames, dtype=bool)
    for ev in events:
        start_f = int(ev["onset"] * sr / hop_length)
        end_f = min(n_frames, int(ev["offset"] * sr / hop_length) + 1)
        covered[start_f:end_f] = True
    if not covered.any():
        return 0.0
    return float(probs[covered].mean())


def process_clip(model, extractor, audio_loader, record, args, device):
    """Returns an accepted pseudo-labeled record, or None if rejected."""
    clip_id = record["clip_id"]
    has_tags = len(record.get("noise_tags", [])) > 0

    try:
        audio, sr = audio_loader.load_audio(clip_id)
    except Exception as e:
        return None, f"audio_load_failed: {e}"

    duration = len(audio) / sr
    try:
        feats = extractor.extract(audio)
    except Exception as e:
        return None, f"feature_extraction_failed: {e}"

    feats = feats.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(feats).squeeze(0)
        probs = torch.sigmoid(logits)

    events = logits_to_events(
        probs, args.threshold, extractor.hop_length, extractor.sr, duration,
        med=args.median_filter, merge_gap=args.merge_gap, min_duration=args.min_duration,
    )
    probs_np = probs.cpu().numpy()
    model_says_positive = len(events) > 0

    # --- agreement rules ---
    if has_tags and model_says_positive:
        conf = mean_confidence_on_mask(probs_np, events, extractor.hop_length, extractor.sr)
        if conf < args.confidence:
            return None, f"low_confidence_positive ({conf:.2f} < {args.confidence})"
        return {
            "clip_id": clip_id, "tier": "bronze_pseudo",
            "duration": duration, "events": events,
        }, "accepted_positive"

    if (not has_tags) and (not model_says_positive):
        return {
            "clip_id": clip_id, "tier": "bronze_pseudo",
            "duration": duration, "events": [],
        }, "accepted_negative"

    return None, "tag_model_disagreement"


def main():
    args = parse_args()
    device = torch.device(args.device)

    bronze_records = load_bronze_records(args.unified_jsonl, args.max_clips)
    print(f"Loaded {len(bronze_records)} Bronze-tier records from {args.unified_jsonl}")
    if not bronze_records:
        print("No Bronze records found -- check unified.jsonl / data_prep.py output.")
        return

    model = CRNNNoiseDetector(n_mels=64)
    model = load_checkpoint(model, args.model, device)
    extractor = LogMelExtractor(sr=16000, n_mels=64, n_fft=1024, hop_length=512)
    audio_loader = AudioLoader(cache_dir=args.cache_dir, target_sr=16000)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"accepted_positive": 0, "accepted_negative": 0,
              "tag_model_disagreement": 0, "low_confidence_positive": 0,
              "audio_load_failed": 0, "feature_extraction_failed": 0}

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, record in enumerate(bronze_records):
            result, reason = process_clip(model, extractor, audio_loader, record, args, device)
            key = reason.split(" ")[0].split("(")[0]
            stats[key] = stats.get(key, 0) + 1
            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (i + 1) % 500 == 0:
                print(f"[{i + 1}/{len(bronze_records)}] processed...")

    total_accepted = stats["accepted_positive"] + stats["accepted_negative"]
    print("=" * 60)
    print(f"Bronze clips scanned     : {len(bronze_records)}")
    print(f"Accepted (positive)      : {stats['accepted_positive']}")
    print(f"Accepted (negative)      : {stats['accepted_negative']}")
    print(f"Total accepted           : {total_accepted} ({total_accepted / len(bronze_records) * 100:.1f}%)")
    print(f"Rejected - disagreement  : {stats['tag_model_disagreement']}")
    print(f"Rejected - low confidence: {stats['low_confidence_positive']}")
    print(f"Rejected - load/feat err : {stats['audio_load_failed'] + stats['feature_extraction_failed']}")
    print(f"Written to               : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
