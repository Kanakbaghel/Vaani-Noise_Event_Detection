import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import VaaniNoiseDataset, collate_fn_vaani
from src.features import LogMelExtractor
from src.model import CRNNNoiseDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline inference for Vaani Noise Event Detection")
    parser.add_argument("--model", type=str, default="models/baseline_crnn.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--input-jsonl", type=str, default="data/processed/val_split.jsonl",
                        help="Path to validation JSONL split")
    parser.add_argument("--cache-dir", type=str, default="data/cache/audio_clips",
                        help="Directory for cached audio files")
    parser.add_argument("--out", type=str, default="submissions/predictions.jsonl",
                        help="Output predictions JSONL path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for event detection")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device (cpu or cuda)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Process at most N samples (for quick testing)")
    parser.add_argument("--cached-first", action=argparse.BooleanOptionalAction, default=False,
                        help="Prioritize cached audio clips (faster for smoke tests)")
    parser.add_argument("--tier-filter", type=str, nargs="*",
                        help="Optional list of tiers to filter (e.g., gold silver)")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please train the model first.")
    ckpt = torch.load(ckpt_path, map_location=device)
    # Support both raw state_dict and dict containing 'model_state_dict'
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def frames_to_seconds(start_frame: int, end_frame: int, hop_length: int, sr: int, duration: float) -> (float, float):
    onset = round(start_frame * hop_length / sr, 3)
    offset = round(min(end_frame * hop_length / sr, duration), 3)
    return onset, offset


def logits_to_events(probs: torch.Tensor, threshold: float, hop_length: int, sr: int, duration: float) -> List[Dict]:
    mask = (probs >= threshold).cpu().numpy().astype(np.uint8)
    events = []
    if mask.sum() == 0:
        return events
    # Find start/end indices of consecutive 1s
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        onset, offset = frames_to_seconds(s, e, hop_length, sr, duration)
        if offset > onset:
            events.append({"onset": onset, "offset": offset})
    return events


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load model
    model = CRNNNoiseDetector(n_mels=64)
    model = load_checkpoint(model, args.model, device)

    # Dataset & feature extractor
    dataset = VaaniNoiseDataset(
        jsonl_path=args.input_jsonl,
        cache_dir=args.cache_dir,
        target_sr=16000,
        tier_filter=args.tier_filter,
        transform=None,
    )
    # Optional: prioritize cached clips to avoid HF streaming latency
    if args.cached_first:
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            cached_files = set(os.listdir(cache_path))
            cached_samples = []
            uncached_samples = []
            for s in dataset.samples:
                fname = f"{s['clip_id']}.wav"
                if fname in cached_files:
                    cached_samples.append(s)
                else:
                    uncached_samples.append(s)
            dataset.samples = cached_samples + uncached_samples
            if cached_samples:
                print(f"Prioritized {len(cached_samples)} locally cached clips.")

    if args.max_samples:
        dataset.samples = dataset.samples[: args.max_samples]
    extractor = LogMelExtractor(sr=16000, n_mels=64, n_fft=1024, hop_length=512)

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    processed = 0
    total_events = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            clip_id = sample["clip_id"]
            audio = sample["audio"].numpy()
            sr = sample["sample_rate"]
            duration = float(sample.get("duration", len(audio) / sr))
            # Feature extraction
            try:
                feats = extractor.extract(audio)
            except Exception as e:
                print(f"[Warning] Feature extraction failed for {clip_id}: {e}")
                events = []
                fout.write(json.dumps({"clip_id": clip_id, "events": events}) + "\n")
                continue
            feats = feats.unsqueeze(0).to(device)  # [1, T, n_mels]
            with torch.no_grad():
                logits = model(feats).squeeze(0)  # [T]
                probs = torch.sigmoid(logits)
            events = logits_to_events(probs, args.threshold, extractor.hop_length, extractor.sr, duration)
            total_events += len(events)
            fout.write(json.dumps({"clip_id": clip_id, "events": events}) + "\n")
            processed += 1
            if args.max_samples and processed >= args.max_samples:
                break
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Inference completed: {processed} clips processed in {elapsed:.2f}s")
    print(f"Total events detected: {total_events}")
    print(f"Predictions written to {out_path}")

if __name__ == "__main__":
    main()
