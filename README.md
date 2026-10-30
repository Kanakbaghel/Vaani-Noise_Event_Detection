<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,10,24,30&height=220&section=header&text=Datathlon@IndoML%202026&fontSize=48&fontColor=ffffff&animation=twinkling&fontAlignY=38&desc=Track%201:%20Noise%20Event%20Detection%20|%20Track%202:%20Noise%20Event%20Removal&descSize=16&descAlignY=58&descAlign=50" width="100%" />
</p>

Team **Deadlock**'s submission for **Datathon@IndoML 2026** — competing in both tracks:
- **Track 1 — Noise Event Detection:** find real-world noise events (vehicle horns, dogs barking, doorbells, kitchen appliances, etc.) in Indic speech recordings and output precise onset/offset timestamps.
- **Track 2 — Noise Event Removal:** given the noise event timestamps (from Track 1), suppress the noise while preserving the underlying speech.

Both use the [Vaani corpus](https://arxiv.org/abs/2603.28714), a large-scale, multi-language Indian speech dataset.

- **Competition:** [indoml.in/datathon](https://indoml.in/datathon)
- **Track 1 on Codabench:** [codabench.org/competitions/17825](https://www.codabench.org/competitions/17825/)
- **Track 2 on Codabench:** [codabench.org/competitions/17835](https://www.codabench.org/competitions/17835/)
- **Final code & report submission deadline:** October 30, 2026, 23:59 IST

### Current status (as of Oct 30, 2026)

| Track | Status | Result |
|---|---|---|
| Track 1 — Detection | ✅ Submitted (Phase 2, final) | **Combined 0.9221** (Event F1 0.3014, Segment Dice 0.6206), rank **51 / 188** on the public leaderboard |
| Track 2 — Removal | ⚠️ Model trained and validated locally; no leaderboard submission | Local held-out validation: SI-SDR −0.38 → +13.21 dB, Combined +11.95. A submission attempt on the final deadline day failed (Codabench server error, phase already closed), so no scored entry exists |

---

## Team

| Name | Role |
|---|---|
| Kanak Baghel (Team Lead) | Data pipeline, coordination, submission tracking (both tracks), **Track 2: model training, fine-tuning, inference/packaging pipeline (Kaggle)**, Track 2 local eval script, Bronze-tier weak supervision, reports |
| Shubham Warkade | Track 1 baseline (CRNN, 9,426-Gold, current submission) + Wav2Vec2 experiments |
| Ankit Kushwaha | Team member |
| Rithish K | Team member |

---

## Problem Overview

- **Task:** detect noise events in speech clips and output onset/offset timestamps per event.
- **Training data (~150h):** three annotation tiers of varying quality —
  - 🥇 **Gold** (~20h) — verified, precise timestamps
  - 🥈 **Silver** (~100h) — timestamps present, not verified
  - 🥉 **Bronze** (~30h) — tag-only, no timestamps
- **Test data (11h, withheld):** 7h natural (real-world) + 4h synthetic clips, scored together.
- **Metric:** `Combined = Event-based F1 (±20% tolerance) + Segment-level Dice` (max 2.0).

Full task/eval details are in [`reports/`](./reports).

---

## Environment & Hardware

**Track 1 (CRNN):**
- **Python version:** 3.12.7
- **CUDA version:** 12.4 (PyTorch 2.6.0+cu124)
- **GPU:** NVIDIA RTX 4050 Laptop GPU, 6GB VRAM
- **OS:** Windows 11
- **Pretrained model dependencies:** `facebook/wav2vec2-base` (Hugging Face) — used only in the Wav2Vec2 experiment, not in the submitted CRNN model
- **Other system dependencies:** ffmpeg (via `soundfile`/`librosa` backends), standard Python 3 build tools

**Track 2 (Removal):**
- **Platform:** Kaggle GPU notebook, Python 3.13 (exact accelerator model was not recorded)
- **Pretrained model dependencies:** `ARTPARK-IISc/SraVaani-1.0` (Hugging Face) — frozen encoder and mandated ASR

---

## Repo Structure

```
Vaani-Noise_Event_Detection/
├── configs/
│   └── baseline.yaml              # CRNN training config (filled)
├── data/
│   ├── raw/                       # downloaded HF parquet shards (gitignored)
│   ├── processed/                 # unified + split JSONL outputs
│   └── cache/
│       └── audio_clips/           # cached audio files (gitignored)
├── docs/
│   ├── audio_loading_guide.md         # audio loading technical guide
│   ├── AUDIO_LOADING_QUICKSTART.md    # quick start guide
│   ├── STABILITY_CHECK_RESULTS.md     # data stability analysis
│   ├── system_description_draft.md    # working notes behind the final report
│   ├── technical_report.tex           # final 3-page IEEE report (Track 1) — source
│   ├── technical_report_track1.pdf    # final 3-page IEEE report (Track 1) — compiled
│   ├── technical_report_track2.tex    # Track 2 report — source
│   └── technical_report_track2.pdf    # Track 2 report — compiled
├── reports/
│   ├── eda_*.png                  # EDA visualizations
│   └── system_description.md      # (superseded by docs/technical_report.tex)
├── notebooks/
│   ├── detection.ipynb
│   └── removal.ipynb
├── src/
│   ├── data_prep.py               # Download + unify Gold/Silver/Bronze
│   ├── split_data.py              # Stratified train/val split
│   ├── eda.py                     # Duration/language/category distribution
│   ├── audio_loader.py            # Lazy audio loading with caching
│   ├── dataset.py                 # PyTorch Dataset wrapper
│   ├── features.py                # Log-mel feature extraction
│   ├── model.py                   # CRNNNoiseDetector (submitted model)
│   ├── advanced_model.py          # Wav2Vec2 + BiGRU detector (experiment)
│   ├── train.py                   # CRNN training loop
│   ├── train_advanced.py          # Wav2Vec2 training loop
│   ├── infer.py                   # Inference -> predictions.jsonl (Track 1)
│   ├── eval_local.py              # Track 1 local Event-F1 + Dice scorer
│   ├── eval_local_removal.py      # Track 2 local SI-SDR + Delta-WER scorer
│   └── bronze_weak_supervision.py # Bronze-tier pseudo-labeling (designed, not run to completion)
├── scripts/
│   ├── cache_gold_subset.py
│   ├── sweep_decoding.py
│   └── sweep_fine_postprocess.py
├── track2/                        # Track 2 (noise removal): training + inference code
│   ├── config.py  data.py  model.py  losses.py  train.py
│   ├── inference.py  evaluate.py  main.py
│   ├── train_track2.py            # CLI: training entry point
│   ├── slim_checkpoint.py         # strips the frozen encoder from the checkpoint
│   ├── infer_track2.py            # CLI: enhance + transcribe + package submission ZIP
│   ├── requirements_train.txt
│   └── requirements_inference.txt
├── tests/                         # setup/stability verification scripts
├── checkpoints/
│   ├── best.pt                    # Track 1 final checkpoint (CRNN, Gold)
│   └── t2_model_real_slim.pt      # Track 2 final checkpoint (mask network only, fine-tuned)
├── submissions/                   # packaged submission.zip files (gitignored)
├── .env.example
├── .gitignore
├── requirements_train.txt         # full training environment
└── requirements_inference.txt     # minimal inference-only environment
```

---

## Setup

1. Clone the repo and create a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements_train.txt      # full training environment
   # OR, for inference only:
   pip install -r requirements_inference.txt
   ```

2. Copy `.env.example` to `.env` and add **your own** Hugging Face token (read access is enough — do not share tokens between team members):
   ```
   HF_TOKEN=your_token_here
   ```

3. Download and unify the dataset:
   ```bash
   python src/data_prep.py --out data/processed/unified.jsonl
   ```

4. Create the train/val split:
   ```bash
   python src/split_data.py
   ```

5. Run EDA:
   ```bash
   python src/eda.py
   ```
   Outputs stats to the terminal and saves plots to `reports/`.

6. **Test the audio loading system:**
   ```bash
   python tests/verify_setup.py
   ```
   See [`docs/AUDIO_LOADING_QUICKSTART.md`](docs/AUDIO_LOADING_QUICKSTART.md) for usage details.

---

## Training

```bash
python src/train.py --tier-filter gold --max-samples 9426 --epochs 5 --batch-size 4 --save-path checkpoints/best.pt
```
CRNN trained on the Gold tier (9,426 clips), 5 epochs, Adam (lr 1e-3),
`BCEWithLogitsLoss`. See `configs/baseline.yaml` for the full configuration
and `docs/technical_report.tex` (Section III) for the architecture and
decoding details.

## Inference

```bash
python src/infer.py --model checkpoints/best.pt --out submissions/predictions.jsonl --create-zip
```
Accepts a Codabench `input_data` directory or `.zip` directly. Produces
`predictions.jsonl` and, with `--create-zip`, a ready-to-upload
`submission_track1.zip`.

**Input:** directory or zip of `.wav`/`.flac`/`.mp3`/`.ogg` clips.
**Output:** `predictions.jsonl`, one line per clip:
```json
{"clip_id": "vaani_eval_001", "events": [{"onset": 1.24, "offset": 3.81}]}
```

**Submission limits (Track 1):** 5/day, 100 total.

### Track 2 (Removal) — training and inference

Code lives in [`track2/`](track2/). It was developed and run in a Kaggle GPU notebook
(Python 3.13); paths are configurable through environment variables
(`T2_WORK` = working dir, `T2_VAL_DIR` = organizer validation folder).

**Method (summary).** A BiGRU magnitude-mask network on the STFT of the noisy clip
(n_fft 512, hop 128, mixture phase reused), conditioned by FiLM at two depths on the frozen
`ARTPARK-IISc/SraVaani-1.0` encoder, trained with a time-domain SI-SDR loss. Stage 1 trains on
simulated mixtures (clean speech spans + isolated noise-event spans from the Gold tier, event
SNR −5…12 dB); stage 2 fine-tunes on the organizers' validation (noisy, clean) pairs with
15% held out. Output stage: blend with the input, RMS restoration, peak guard. The architecture and
pipeline follow the organizers' published v-1 Track 2 baseline; see the report for exactly what we ran.

```bash
pip install -r track2/requirements_train.txt        # training
pip install -r track2/requirements_inference.txt    # inference only
export HF_TOKEN=your_own_read_token                 # never commit tokens

# Training (as run for the reported results: budget tiny, fine-tune on)
export T2_VAL_DIR=/path/to/indoml-validation T2_WORK=track2_work
python track2/train_track2.py --budget tiny      # writes t2_model.pt and t2_model_real.pt
# t2_model_real.pt also stores the frozen SraVaani encoder (~800 MB+); keep only the trained part:
python track2/slim_checkpoint.py track2_work/t2_model_real.pt checkpoints/t2_model_real_slim.pt

# Inference: enhance -> SraVaani transcripts -> submission_track2.zip
python track2/infer_track2.py --checkpoint checkpoints/t2_model_real_slim.pt \
       --test-dir /path/to/test/audio --out-dir track2_out --blend 1.0
```
Input: a folder of audio clips. Output: `track2_out/submission_track2.zip` containing one enhanced
16 kHz mono PCM-16 WAV per clip plus `transcripts.jsonl` at the ZIP root.
The pretrained SraVaani-1.0 weights are downloaded from Hugging Face on first use.

**Track 2 environment:** Kaggle GPU notebook, Python 3.13 (see "Environment & Hardware").

### Track 2 (Removal) submission format

ZIP containing enhanced WAVs (16kHz mono PCM16, one per clip, named `<clip_id>.wav`) plus a single `transcripts.jsonl` at the root, all from running the mandated `ARTPARK-IISc/SraVaani-1.0` ASR on the enhanced audio:

```json
{"clip_id": "vaani_eval_001", "text": "asr transcript of the enhanced audio"}
```

Sanity-check locally before uploading:
```bash
python src/eval_local_removal.py \
  --enhanced-dir submissions/enhanced/ \
  --transcripts submissions/transcripts.jsonl \
  --val-metadata data/processed/val_metadata.jsonl \
  --noisy-asr data/processed/val_noisy_asr.jsonl
```

**Submission limits (Track 2):** 3/day, 50 total. No successful submission was made for this track — see "Current status" above.

---

## Results

| Track | Metric | Local Validation | Leaderboard |
|---|---|---|---|
| Track 1 — Detection (CRNN, submitted) | Combined (Event-F1 + Dice) | 0.8418 | **0.9221**, rank 51/188 |
| Track 1 — CRNN Gold + 5k Silver (not submitted) | Combined | 0.8436 | — |
| Track 1 — CRNN Gold + 10k Silver (not submitted) | Combined | 0.7906 | — |
| Track 1 — Wav2Vec2 (not submitted) | Combined | 0.7408 | — |
| Track 2 — Removal (mask net + SraVaani FiLM, fine-tuned) | Combined (SI-SDR + 100·ΔWER), held-out 15% of validation | **+11.95** (SI-SDR +13.21, ΔWER −1.26%) | not submitted |

Full writeups: Track 1 — [`docs/technical_report.tex`](docs/technical_report.tex) /
[`docs/technical_report_track1.pdf`](docs/technical_report_track1.pdf); Track 2 —
[`docs/technical_report_track2.tex`](docs/technical_report_track2.tex) /
[`docs/technical_report_track2.pdf`](docs/technical_report_track2.pdf).

---

## Status

**Track 1 — Detection**
- [x] Data pipeline, EDA, audio loading/caching
- [x] CRNN baseline, trained and submitted — **0.9221 on leaderboard, rank 51/188**
- [x] Local eval script (`eval_local.py`)
- [x] Wav2Vec2 experiment (0.7408 val, below CRNN — documented, not submitted)
- [x] Technical report (3-page IEEE format)
- [x] Final checkpoint committed to repo (`checkpoints/best.pt`)
- [x] CRNN retrain with Silver tier: +5k Silver 0.8436 local (marginal gain), +10k Silver 0.7906 (worse) — Gold-only checkpoint kept as final
- [ ] Bronze weak-supervision (designed, not run to completion)

**Track 2 — Removal**
- [x] Registered for the competition
- [x] Local eval script (`eval_local_removal.py`)
- [x] Mask network trained on simulated mixtures, fine-tuned on validation pairs (held-out SI-SDR +13.21 dB)
- [x] Inference + mandated SraVaani ASR + submission packaging (`track2/infer_track2.py`)
- [ ] Successful leaderboard submission — none made (Codabench server error on final attempt)

---

## Notes

- Never commit `.env` or any access tokens — they're gitignored, keep it that way.
- The dataset's image files are not needed for this track (audio + metadata only).
- `*.pt` / `*.pth` / `*.ckpt` are gitignored by default, with explicit exceptions for `checkpoints/best.pt` and `checkpoints/t2_model_real_slim.pt` so the final checkpoints are tracked.
