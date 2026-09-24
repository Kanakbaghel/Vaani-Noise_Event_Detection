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
- **Phase 1 (Half-Marathon) deadline:** September 22, 2026 (extended from Sep 20 due to Codabench downtime)
- **Final deadline:** October 17, 2026 (extended from Oct 15)

### Current status (as of Sep 23, 2026)

| Track | Status | Result |
|---|---|---|
| Track 1 — Detection | ✅ Phase 1 submitted | **Score 0.9221**, rank **44 / 78** on the leaderboard (not yet in the prize zone — top 8) |
| Track 2 — Removal | ⚠️ Registered, Phase 1 window missed (0 submissions) | Phase 2 (final round) is open until Oct 17 — work starting now |

Track 1 work continues past Phase 1 since the leaderboard does not reset — every improvement before Oct 17 counts. Track 2 is new work, not a restart of Track 1.

---

## Team

| Name | Role |
|---|---|
| Kanak Baghel (Team Lead) | Data pipeline, coordination, submission tracking (both tracks), Track 2 local eval script, Bronze-tier weak supervision, system description report |
| Shubham Warkade | Track 1 baseline (CRNN, 9,426-Gold, current submission) + Wav2Vec2 experiments |
| Ankit Kushwaha | Track 2 — trained denoising model (spectrogram-based enhancement) |
| Rithish K | Track 2 — signal-processing baseline + mandated ASR (SraVaani-1.0) transcript pipeline |

Soubhik was originally assigned the local eval script and Bronze-tier weak supervision; those tasks were absorbed by Kanak. Full task breakdown with dates lives in the team's shared planning spreadsheet, not in this repo.

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

## Repo Structure

```
Vaani-Noise_Event_Detection/
├── configs/
│   └── baseline.yaml              # model/training config
├── data/
│   ├── raw/                       # downloaded HF parquet shards (gitignored)
│   ├── processed/                 # unified + split JSONL outputs
│   └── cache/
│       └── audio_clips/           # cached audio files (gitignored)
├── docs/                          # Documentation
│   ├── audio_loading_guide.md     # Full technical guide for audio loading
│   ├── AUDIO_LOADING_QUICKSTART.md # Quick start guide
│   ├── TESTING_GUIDE.md           # Testing instructions
│   ├── SYSTEM_STATUS.md           # Current system status
│   ├── STABILITY_CHECK_RESULTS.md # Data stability analysis
│   ├── SETUP_SUMMARY.md           # Setup summary
│   └── FINAL_VERDICT.md           # Production readiness assessment
├── reports/
│   ├── eda_*.png                  # EDA visualizations
│   └── system_description.md      # Top-5 system write-up (WIP)
├── src/
│   ├── data_prep.py               # Download + unify Gold/Silver/Bronze
│   ├── split_data.py              # Stratified train/val split
│   ├── eda.py                     # Duration/language/category distribution
│   ├── audio_loader.py            # Lazy audio loading with caching ✓
│   ├── dataset.py                 # PyTorch Dataset wrapper ✓
│   ├── train.py                   # Training loop (WIP)
│   ├── infer.py                   # Inference -> predictions.jsonl ✓
│   ├── eval_local.py              # Track 1 local Event-F1 + Dice scorer ✓
│   └── eval_local_removal.py      # Track 2 local SI-SDR + Delta-WER scorer ✓
├── tests/                         # Test scripts
│   ├── verify_setup.py            # Setup verification
│   ├── analyze_existing_data.py   # Data structure analysis
│   ├── download_one_audio.py      # Single audio download test
│   ├── quick_test.py              # Quick system test
│   ├── test_audio_download.py     # Comprehensive audio test
│   ├── check_dataset_stability.py # HF dataset stability check
│   └── quick_stability_check.py   # Fast stability verification
├── submissions/                   # Packaged submission.zip files
├── .env.example                   # Environment template
├── .gitignore
└── requirements.txt
```

---

## Setup

1. Clone the repo and create a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
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

## Submission Format

Predictions must be a `predictions.jsonl` file zipped at the archive root:

```json
{"clip_id": "vaani_eval_001", "events": [{"onset": 1.24, "offset": 3.81}]}
{"clip_id": "vaani_eval_002", "events": []}
```

Use `src/infer.py` to generate this from a trained model, then package with:
```bash
python src/infer.py --model checkpoints/best.pt --out submissions/predictions.jsonl
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

**Submission limits (Track 2):** 3/day, 50 total (Phase 1 window already closed for us; Phase 2 is open until Oct 17).

Track usage in the team submission log before submitting on Codabench, for either track.

---

## Status

**Track 1 — Detection**
- [x] Repo scaffolding
- [x] Data download + Gold/Silver/Bronze unification (`data_prep.py`)
- [x] Train/val split (`split_data.py`)
- [x] EDA (`eda.py`)
- [x] Lazy audio loading system with caching (`audio_loader.py`, `dataset.py`)
- [x] Baseline model (CRNN) + first submission — **0.9221 on leaderboard, rank 44/78**
- [x] Local eval script (`eval_local.py`)
- [x] Wav2Vec2 experiment (0.7408 val, below CRNN — kept for future tuning)
- [ ] CRNN retrain with Silver tier + post-processing sweep (target: beat 0.8418 val)
- [ ] Full 9k-Gold Wav2Vec2 run + possible ensemble
- [ ] Bronze weak-supervision (novelty angle)
- [ ] System description report (if top-5)

**Track 2 — Removal**
- [x] Registered for the competition
- [x] Local eval script (`eval_local_removal.py`)
- [ ] Signal-processing baseline (spectral gating on Track 1's predicted noise segments)
- [ ] Mandated ASR (SraVaani-1.0) transcript pipeline
- [ ] First submission
- [ ] Trained denoising model (spectrogram-based enhancement)

---

## Notes

- Never commit `.env` or any access tokens — they're gitignored, keep it that way.
- The dataset's image files are not needed for this track (audio + metadata only).
