"""
train_advanced.py
Advanced fine-tuning pipeline for Vaani Noise Event Detection.

Connects:
- VaaniNoiseDataset (Gold + Silver) + collate_fn_vaani  (unchanged, shared)
- Wav2Vec2NoiseDetector (pretrained encoder + BiGRU detection head)
- Frame-level binary targets aligned to Wav2Vec2's 320-sample stride
- BCEWithLogitsLoss + AdamW optimizer

Goal: beat the baseline CRNN on onset/offset (frame-level) accuracy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import VaaniNoiseDataset, collate_fn_vaani
from src.advanced_model import Wav2Vec2NoiseDetector, WAV2VEC2_STRIDE, WAV2VEC2_SR

# Windows-safe console output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced Wav2Vec2 fine-tuning for Vaani Noise Event Detection"
    )
    parser.add_argument("--train-jsonl", type=str, default="data/processed/train_split.jsonl")
    parser.add_argument("--cache-dir", type=str, default="data/cache/audio_clips")
    parser.add_argument("--save-path", type=str, default="models/advanced_wav2vec2.pt")
    parser.add_argument("--pretrained", type=str, default="facebook/wav2vec2-base",
                        help="HuggingFace Wav2Vec2 encoder id")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Small default: Wav2Vec2 is memory-heavy (keep low on 4GB GPUs)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max training samples (default 100 for fast iteration)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cached-first", action=argparse.BooleanOptionalAction, default=True,
                        help="Prioritize locally cached clips to avoid HF streaming delays")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_targets(
    batch: dict,
    stride: int = WAV2VEC2_STRIDE,
    sr: int = WAV2VEC2_SR,
) -> Tuple[torch.Tensor, int]:
    """
    Convert event annotations into per-frame binary targets aligned to the
    Wav2Vec2 output grid (one frame per `stride` input samples).

    Uses the padded audio length in the batch to size the frame axis, so the
    targets line up with the encoder output for the (equally padded) batch.

    Returns:
        targets: FloatTensor [B, T_frames]
        num_frames: int
    """
    audio_padded = batch["audio"]      # [B, max_samples]
    durations = batch["durations"]     # [B]
    events_batch = batch["events"]     # List[List[Dict]]
    clip_ids = batch["clip_ids"]       # List[str]

    batch_size, max_samples = audio_padded.shape
    # Frames the encoder emits for the padded length: floor((L - stride)/stride)+1,
    # approximated by L // stride which matches Wav2Vec2 conv output for base.
    num_frames = max(1, max_samples // stride)

    targets = torch.zeros(batch_size, num_frames, dtype=torch.float32)

    for i in range(batch_size):
        sample_events = events_batch[i]
        if not isinstance(sample_events, list):
            print(f"[Warning] Unexpected events format in {clip_ids[i]}: {type(sample_events)}")
            continue
        for ev in sample_events:
            if not isinstance(ev, dict):
                print(f"[Warning] Malformed event (not dict) in {clip_ids[i]}: {ev}; skipping.")
                continue
            onset = ev.get("onset")
            offset = ev.get("offset")
            if onset is None or offset is None:
                print(f"[Warning] Event missing onset/offset in {clip_ids[i]}: {ev}; skipping.")
                continue
            try:
                onset = float(onset)
                offset = float(offset)
            except (ValueError, TypeError):
                print(f"[Warning] Bad event timestamps in {clip_ids[i]}: {ev}; skipping.")
                continue
            if onset < 0 or offset < onset:
                print(f"[Warning] Invalid interval [{onset}, {offset}] in {clip_ids[i]}; skipping.")
                continue
            start_frame = max(0, int(np.floor(onset * sr / stride)))
            end_frame = min(num_frames, int(np.ceil(offset * sr / stride)))
            if end_frame > start_frame:
                targets[i, start_frame:end_frame] = 1.0
            elif start_frame < num_frames:
                targets[i, start_frame:start_frame + 1] = 1.0

    return targets, num_frames


def align_length(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim logits/targets to a common frame length (off-by-one guard)."""
    T = min(logits.shape[1], targets.shape[1])
    return logits[:, :T], targets[:, :T]


def train():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("VAANI NOISE EVENT DETECTION - ADVANCED (Wav2Vec2) TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Encoder: {args.pretrained}")

    # 1. Dataset (Gold + Silver, raw waveforms -> transform=None)
    dataset = VaaniNoiseDataset(
        jsonl_path=args.train_jsonl,
        cache_dir=args.cache_dir,
        target_sr=WAV2VEC2_SR,
        tier_filter=["gold", "silver"],
        transform=None,
    )

    # Prioritize cached clips to avoid streaming latency during smoke tests
    if args.cached_first:
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            cached_files = set(os.listdir(cache_path))
            cached, uncached = [], []
            for s in dataset.samples:
                (cached if f"{s['clip_id']}.wav" in cached_files else uncached).append(s)
            dataset.samples = cached + uncached
            if cached:
                print(f"Prioritized {len(cached)} locally cached clips at start of dataset.")

    if args.max_samples is not None and args.max_samples > 0:
        dataset.samples = dataset.samples[:args.max_samples]

    print(f"Number of training samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    if len(dataset) == 0:
        raise RuntimeError("No training samples found matching the criteria.")

    # 2. DataLoader (shared collate_fn pads raw waveforms)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(len(dataset) > args.batch_size),
        collate_fn=collate_fn_vaani,
        num_workers=args.num_workers,
    )

    # 3. Model, Loss, Optimizer
    model = Wav2Vec2NoiseDetector(pretrained_name=args.pretrained).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 4. Training loop
    first_batch_logged = False
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            waveforms = batch["audio"].to(device)          # [B, max_samples]
            targets, _ = build_targets(batch)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(waveforms)                      # [B, T]
            logits, targets = align_length(logits, targets)

            if not first_batch_logged:
                print(f"Waveform batch: {list(waveforms.shape)} -> logits {list(logits.shape)}")
                first_batch_logged = True

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {epoch_loss:.4f}")

    # 5. Save checkpoint
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pretrained": args.pretrained,
            "stride": WAV2VEC2_STRIDE,
            "sample_rate": WAV2VEC2_SR,
            "epochs": args.epochs,
            "final_loss": epoch_loss,
            "args": vars(args),
        },
        save_path,
    )
    print(f"Model checkpoint saved successfully to {save_path}")
    print("Advanced training completed successfully.")


if __name__ == "__main__":
    train()
