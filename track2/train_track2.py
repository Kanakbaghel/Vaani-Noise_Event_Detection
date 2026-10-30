"""train_track2.py - Track 2 training entry point (simulation stage + real-pair fine-tune).

Usage:
    export HF_TOKEN=...                 # your own read token (never commit it)
    export T2_VAL_DIR=/path/to/indoml-validation
    export T2_WORK=track2_work          # checkpoints land here (t2_model.pt, t2_model_real.pt)
    python track2/train_track2.py --budget tiny
The reported results used --budget tiny with fine-tuning on. The final checkpoint is
t2_model_real.pt (best held-out absolute SI-SDR); copy it to checkpoints/.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="tiny", choices=["tiny", "fast", "full"])
    ap.add_argument("--no-finetune", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from main import run
    best_blend = run(budget=args.budget, hf_token=os.environ.get("HF_TOKEN"),
                     do_finetune=not args.no_finetune, do_submission=False)
    print("best blend:", best_blend)


if __name__ == "__main__":
    main()
