"""
eval_local_removal.py
Local scorer for Vaani Noise Event Removal (IndoML 2026, Track 2 / competition 17835).

Reproduces the official Codabench metric locally so we can sanity-check a submission
before spending one of our limited daily uploads:

    Combined = SI-SDR(synthetic subset) + 100 * Delta-WER(full test set)

This mirrors the organizers' reference code exactly (same si_sdr formula, same
delta_wer normalization + pooled jiwer.wer), so a local score here should closely
match what Codabench reports.

Usage:
    python src/eval_local_removal.py \
        --enhanced-dir submissions/enhanced/ \
        --transcripts submissions/transcripts.jsonl \
        --val-metadata data/processed/val_metadata.jsonl \
        --noisy-asr data/processed/val_noisy_asr.jsonl

Required inputs
----------------
1. Enhanced WAVs: one file per validation clip, named <clip_id>.wav, in --enhanced-dir.
   Must be 16kHz mono PCM16, same duration as the original.

2. --transcripts: our transcripts.jsonl, ASR output from the enhanced audio
   (SraVaani-1.0), one line per clip:
       {"clip_id": "...", "text": "..."}

3. --val-metadata: validation set ground truth, one line per clip:
       {"clip_id": "...", "syntheticData": true/false,
        "reference_wav": "path/to/clean_reference.wav" (synthetic clips only),
        "ground_truth_text": "..."}
   (Field names are adapted to whatever the downloaded Codabench validation
   metadata actually uses -- see NOTE below if they differ.)

4. --noisy-asr: the ASR transcript of the ORIGINAL noisy audio (the fixed
   baseline WER_noisy is computed from this), one line per clip:
       {"clip_id": "...", "text": "..."}
   If you don't have this cached yet, pass --skip-delta-wer to get SI-SDR only.

NOTE ON FIELD NAMES: the exact key names in the downloaded validation metadata
file aren't confirmed yet (we haven't pulled it from Codabench's Get Started >
Files section). Field names below are best guesses based on the task page's
"syntheticData" flag; adjust the `FIELD_*` constants at the top of the script
once the real file is in hand -- everything else (SI-SDR math, WER pooling,
text normalization) is copied verbatim from the organizers' published
reference code and should not need changes.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import wer

# Adjust these if the real validation metadata uses different key names.
FIELD_CLIP_ID = "clip_id"
FIELD_SYNTHETIC_FLAG = "syntheticData"
FIELD_REFERENCE_WAV = "reference_wav"
FIELD_GROUND_TRUTH_TEXT = "ground_truth_text"

TAG_RE = re.compile(r"</?[^<>]+>|\[[^\[\]]*\]")


def parse_args():
    p = argparse.ArgumentParser(description="Local evaluator for Vaani Noise Event Removal (Track 2)")
    p.add_argument("--enhanced-dir", type=str, required=True,
                    help="Directory containing enhanced <clip_id>.wav files")
    p.add_argument("--transcripts", type=str, required=True,
                    help="Our transcripts.jsonl (ASR on enhanced audio)")
    p.add_argument("--val-metadata", type=str, required=True,
                    help="Validation set ground-truth metadata JSONL")
    p.add_argument("--noisy-asr", type=str, default=None,
                    help="ASR transcripts of the ORIGINAL noisy audio (for WER_noisy baseline)")
    p.add_argument("--skip-delta-wer", action="store_true",
                    help="Only compute SI-SDR, skip Delta-WER (use if --noisy-asr isn't available yet)")
    p.add_argument("--per-clip", action="store_true",
                    help="Print per-clip SI-SDR for the synthetic subset")
    return p.parse_args()


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec[FIELD_CLIP_ID]] = rec
    return records


def load_text_map(path):
    """clip_id -> text, from a {"clip_id": ..., "text": ...} jsonl."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["clip_id"]] = rec.get("text", "")
    return out


# --------------------------------------------------------------------- #
# SI-SDR — copied from the organizers' reference code
# --------------------------------------------------------------------- #

def si_sdr(reference, enhanced):
    ref = np.asarray(reference, dtype=np.float64)
    enh = np.asarray(enhanced, dtype=np.float64)
    n = min(len(ref), len(enh))
    ref, enh = ref[:n], enh[:n]

    scale = np.dot(enh, ref) / np.dot(ref, ref)
    s_target = scale * ref
    e_noise = enh - s_target
    value = 10.0 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise))
    return float(np.clip(value, -100.0, 100.0))


# --------------------------------------------------------------------- #
# Delta-WER — copied from the organizers' reference code
# --------------------------------------------------------------------- #

def normalize(text):
    s = TAG_RE.sub(" ", text or "")
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.lower().split())


def delta_wer(gt, noisy_asr, submitted, clip_ids):
    refs, noisy, enh = [], [], []
    for cid in clip_ids:
        g = normalize(gt.get(cid, ""))
        if not g:
            continue
        refs.append(g)
        noisy.append(normalize(noisy_asr.get(cid, "")) or "@")
        enh.append(normalize(submitted.get(cid, "")) or "@")
    if not refs:
        return 0.0, 0.0, 0.0
    wer_noisy = wer(refs, noisy)
    wer_enh = wer(refs, enh)
    return wer_noisy - wer_enh, wer_noisy, wer_enh


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    args = parse_args()

    val_meta = load_jsonl(args.val_metadata)
    submitted_text = load_text_map(args.transcripts)
    enhanced_dir = Path(args.enhanced_dir)

    all_clip_ids = list(val_meta.keys())
    synthetic_ids = [cid for cid in all_clip_ids if val_meta[cid].get(FIELD_SYNTHETIC_FLAG)]

    print(f"Total validation clips: {len(all_clip_ids)}")
    print(f"Synthetic subset (SI-SDR scored): {len(synthetic_ids)}")

    # ---- SI-SDR on synthetic subset ---- #
    si_sdr_scores = []
    missing_enhanced = 0
    missing_reference = 0
    per_clip_sisdr = []

    for cid in synthetic_ids:
        enh_path = enhanced_dir / f"{cid}.wav"
        ref_path = val_meta[cid].get(FIELD_REFERENCE_WAV)
        if not enh_path.exists():
            missing_enhanced += 1
            continue
        if not ref_path or not Path(ref_path).exists():
            missing_reference += 1
            continue
        enhanced, _ = sf.read(str(enh_path))
        reference, _ = sf.read(str(ref_path))
        score = si_sdr(reference, enhanced)
        si_sdr_scores.append(score)
        per_clip_sisdr.append((cid, score))

    mean_sisdr = sum(si_sdr_scores) / len(si_sdr_scores) if si_sdr_scores else 0.0

    if args.per_clip:
        print(f"\n{'clip_id':<40} {'SI-SDR (dB)':>12}")
        for cid, score in per_clip_sisdr:
            print(f"{cid:<40} {score:>12.3f}")

    print(f"\nSI-SDR: {mean_sisdr:.3f} dB  "
          f"(missing enhanced: {missing_enhanced}, missing reference: {missing_reference})")

    # ---- Delta-WER on full set ---- #
    combined = mean_sisdr
    if not args.skip_delta_wer and args.noisy_asr:
        noisy_asr_text = load_text_map(args.noisy_asr)
        gt_text = {cid: val_meta[cid].get(FIELD_GROUND_TRUTH_TEXT, "") for cid in all_clip_ids}
        delta, wer_noisy, wer_enh = delta_wer(gt_text, noisy_asr_text, submitted_text, all_clip_ids)
        print(f"WER (noisy):    {wer_noisy:.4f}")
        print(f"WER (enhanced): {wer_enh:.4f}")
        print(f"Delta-WER:      {delta:.4f}  ({delta * 100:.2f}%)")
        combined = mean_sisdr + 100 * delta
    else:
        print("Skipped Delta-WER (pass --noisy-asr to include it in Combined).")

    print("=" * 50)
    print(f"Combined (SI-SDR + 100*DeltaWER): {combined:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
