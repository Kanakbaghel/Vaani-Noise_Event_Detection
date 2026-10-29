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

### Current status (as of Oct 29, 2026)

| Track | Status | Result |
|---|---|---|
| Track 1 — Detection | ✅ Submitted (Phase 2, final) | **Combined 0.9221** (Event F1 0.3014, Segment Dice 0.6206), rank **51 / 188** on the public leaderboard |
| Track 2 — Removal | ⚠️ Code built, no leaderboard submission | A submission attempt on the final deadline day failed due to a Codabench server error after the phase had already closed; no scored entry exists for this track |

---

## Team

| Name | Role |
|---|---|
| Kanak Baghel (Team Lead) | Data pipeline, coordination, submission tracking (both tracks), Track 2 local eval script, Bronze-tier weak supervision, system description report |
| Shubham Warkade | Track 1 baseline (CRNN, 9,426-Gold, current submission) + Wav2Vec2 experiments |
| Ankit Kushwaha | Track 2 — trained denoising model (spectrogram-based enhancement) |
| Rithish K | Track 2 — signal-processing baseline + mandated ASR (SraVaani-1.0) transcript pipeline |

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

- **Python version:** <fill in — e.g. 3.10.x>
- **CUDA version:** <fill in>
- **GPU:** NVIDIA RTX 4050, 6GB VRAM
- **OS:** <fill in — e.g. Windows 11 / Ubuntu 22.04>
- **Pretrained model dependencies:** `facebook/wav2vec2-base` (Hugging Face) — used only in the Wav2Vec2 experiment, not in the submitted CRNN model
- **Other system dependencies:** ffmpeg (via `soundfile`/`librosa` backends), standard Python 3 build tools

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
│   └── technical_report_track1.pdf    # final 3-page IEEE report (Track 1) — compiled
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
├── tests/                         # setup/stability verification scripts
├── checkpoints/                   # final trained checkpoint goes here (checkpoints/best.pt)
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
python src/train.py --config configs/baseline.yaml
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
| Track 1 — Wav2Vec2 (not submitted) | Combined | 0.7408 | — |
| Track 2 — Removal | Combined (SI-SDR + 100·ΔWER) | not completed | not submitted |

Full writeup: [`docs/technical_report.tex`](docs/technical_report.tex) /
[`docs/technical_report_track1.pdf`](docs/technical_report_track1.pdf).

---

## Status

**Track 1 — Detection**
- [x] Data pipeline, EDA, audio loading/caching
- [x] CRNN baseline, trained and submitted — **0.9221 on leaderboard, rank 51/188**
- [x] Local eval script (`eval_local.py`)
- [x] Wav2Vec2 experiment (0.7408 val, below CRNN — documented, not submitted)
- [x] Technical report (3-page IEEE format)
- [ ] Final checkpoint committed to repo (`checkpoints/best.pt`) — pending
- [ ] CRNN retrain with Silver tier + post-processing sweep
- [ ] Bronze weak-supervision (designed, not run to completion)

**Track 2 — Removal**
- [x] Registered for the competition
- [x] Local eval script (`eval_local_removal.py`)
- [x] Denoising model + ASR pipeline built
- [ ] Successful leaderboard submission — none made (Codabench server error on final attempt)

---

## Notes

- Never commit `.env` or any access tokens — they're gitignored, keep it that way.
- The dataset's image files are not needed for this track (audio + metadata only).
- `*.pt` / `*.pth` / `*.ckpt` are gitignored by default, with an explicit exception for `checkpoints/best.pt` so the final submitted checkpoint is tracked.
