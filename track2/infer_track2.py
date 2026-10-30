"""infer_track2.py - Track 2 (Noise Event Removal) inference + submission packaging.

Enhances every clip in --test-dir with the trained mask network, transcribes the
enhanced audio with the mandated ARTPARK-IISc/SraVaani-1.0 ASR, and writes
<out-dir>/submission_track2.zip (enhanced 16 kHz mono PCM-16 WAVs + transcripts.jsonl
at the ZIP root).

Usage:
    export HF_TOKEN=...            # your own read token (never commit it)
    python track2/infer_track2.py \
        --checkpoint checkpoints/t2_model_real_slim.pt \
        --test-dir /path/to/test/audio \
        --out-dir track2_out --blend 1.0
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Track 2 inference + packaging")
    ap.add_argument("--checkpoint", required=True, help="fine-tuned checkpoint (t2_model_real_slim.pt)")
    ap.add_argument("--test-dir", required=True, help="folder with the test audio clips")
    ap.add_argument("--out-dir", default="track2_out", help="working/output directory")
    ap.add_argument("--blend", type=float, default=1.0,
                    help="y = blend*est + (1-blend)*mix; 1.0 was best on held-out validation")
    args = ap.parse_args()

    # config.py reads T2_WORK at import time, so set it before importing the modules.
    os.environ["T2_WORK"] = str(Path(args.out_dir).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import data
    import model as model_mod
    from inference import Transcriber
    from main import write_submission

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[warn] HF_TOKEN not set; relying on cached/anonymous Hugging Face access")

    model, _encoder, _ = model_mod.core_model(token)
    ckpt = torch.load(args.checkpoint, map_location=data.device, weights_only=False)
    # The slim checkpoint omits the frozen SraVaani encoder (it is downloaded from
    # Hugging Face by core_model), so only encoder keys may be missing.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad = [k for k in missing if not k.startswith("enc.")]
    assert not bad and not unexpected, f"checkpoint mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}"
    model.eval()

    transcriber = Transcriber(token)
    write_submission(model, transcriber, args.blend, test_dir=Path(args.test_dir))


if __name__ == "__main__":
    main()
