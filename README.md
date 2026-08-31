<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,10,24,30&height=220&section=header&text=Datathlon@IndoML%202026&fontSize=48&fontColor=ffffff&animation=twinkling&fontAlignY=38&desc=Track%201:%20Noise%20Event%20Detection%20|%20Track%202:%20Noise%20Event%20Removal&descSize=16&descAlignY=58&descAlign=50" width="100%" />
</p>

Team **Deadlock**'s submission for **Datathon@IndoML 2026 — Track 1: Noise Event Detection**.

We're building a system to detect real-world noise events (vehicle horns, dogs barking, doorbells, kitchen appliances, etc.) in Indic speech recordings, with precise onset/offset timestamps — using the [Vaani corpus](https://arxiv.org/abs/2603.28714), a large-scale, multi-language Indian speech dataset.

- **Competition:** [indoml.in/datathon](https://indoml.in/datathon)
- **Track 1 on Codabench:** [codabench.org/competitions/17825](https://www.codabench.org/competitions/17825/)
- **Phase 1 (Half-Marathon) deadline:** September 20, 2026
- **Final deadline:** October 15, 2026

---

## Team

| Name | Role |
|---|---|
| Kanak Baghel (Team Lead) | Data pipeline, annotation-tier unification, EDA, coordination, submission tracking |
| Shubham Warkade | Baseline model (feature extraction, CRNN, first end-to-end submission) |
| Ankit Kushwaha | Advanced modeling (pretrained audio encoders, fine-tuning) — paired with Rithish |
| Rithish K | Advanced modeling (onset/offset boundary prediction) — paired with Ankit |
| Soubhik Shit | Local evaluation script, Bronze-tier weak supervision, natural vs synthetic analysis |

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
│   └── baseline.yaml          # model/training config
├── data/
│   ├── raw/                   # downloaded HF parquet shards (gitignored)
│   └── processed/             # unified + split JSONL outputs (gitignored)
├── notebooks/
│   └── detection.ipynb        # exploratory notebook
├── reports/
│   └── system_description.md  # top-5 system write-up (WIP)
├── src/
│   ├── data_prep.py           # download + unify Gold/Silver/Bronze -> unified.jsonl
│   ├── split_data.py          # stratified train/val split
│   ├── eda.py                 # duration/language/noise-category distribution
│   ├── dataset.py             # PyTorch/HF Dataset wrapper (WIP)
│   ├── train.py                # training loop (WIP)
│   ├── infer.py                # inference -> predictions.jsonl (WIP)
│   └── eval_local.py          # local Event-F1 + Dice scorer (WIP)
├── submissions/                # packaged submission.zip files
├── .env.example
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

**Submission limits:** 5/day, 100 total. Track usage in the team submission log before submitting on Codabench.

---

## Status

- [x] Repo scaffolding
- [x] Data download + Gold/Silver/Bronze unification (`data_prep.py`)
- [ ] Train/val split
- [ ] EDA
- [ ] Baseline model + first submission
- [ ] Advanced modeling
- [ ] Local eval script
- [ ] Bronze weak-supervision
- [ ] System description report (if top-5)

---

## Notes

- Never commit `.env` or any access tokens — they're gitignored, keep it that way.
- The dataset's image files are not needed for this track (audio + metadata only).
