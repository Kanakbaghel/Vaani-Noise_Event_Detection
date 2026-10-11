# System Description — Team Deadlock
### Datathon@IndoML 2026 — Track 1 (Noise Event Detection) & Track 2 (Noise Event Removal)

> **STATUS: DRAFT, in progress.** Sections marked `[TBD]` are placeholders for results
> not yet finalized as of this draft (Oct 10, 2026). Required only if the team places
> in the top 5 of either leaderboard; being built incrementally so it isn't rushed at
> the deadline. Max length for final submission: 4 pages.

**Team:** Deadlock — Kanak Baghel (Lead), Shubham Warkade, Ankit Kushwaha, Rithish K
**Contact:** kanak.bgl704@gmail.com
**Repository:** github.com/Kanakbaghel/Vaani-Noise_Event_Detection

---

## 1. Problem Summary

The Vaani corpus provides large-scale, multi-language Indian speech recordings with
three tiers of noise-event annotation: Gold (~20h, verified onset/offset timestamps),
Silver (~100h, unverified timestamps), and Bronze (~30h, noise-event tags in the
transcript only, no timestamps). We tackle both competition tracks built on this data:

- **Track 1 — Detection:** locate noise events in a recording and output precise
  onset/offset timestamps per event.
- **Track 2 — Removal:** given a recording and its noise-event timestamps (from
  Track 1 or ground truth), suppress the noise while preserving the underlying speech.

---

## 2. Track 1 — Noise Event Detection

### 2.1 Approach

Our primary model is a compact CRNN (convolutional-recurrent neural network) trained
directly on log-mel spectrogram features, predicting a per-frame noise-presence
probability that is thresholded and post-processed into discrete onset/offset events.
We chose this over a large pretrained encoder first because it trains fast on modest
hardware and gave us a strong, submittable baseline early, which we then used as the
foundation for every later experiment (including conditioning Track 2's model).

In parallel we explored a Wav2Vec2-based detector (Section 2.3) and a weak-supervision
extension using the Bronze tier (Section 2.4), to test whether a larger pretrained
encoder or additional (if noisier) data could beat the CRNN.

### 2.2 Architecture — CRNN baseline

- **Input:** 64-band log-mel spectrogram (16kHz audio, 1024-point FFT, 512-sample hop)
- **Backbone:** 2 convolutional blocks (16 -> 32 channels, 3x3 kernels, batch norm,
  ReLU, frequency-axis max-pooling) feeding a single-layer bidirectional LSTM
  (hidden size 64)
- **Head:** linear layer -> per-frame logit -> sigmoid -> noise-event probability
- **Decoding:** probability threshold (0.35) + temporal median filter (window 7) to
  denoise the frame-level mask, followed by minimum-duration (0.15s) and merge-gap
  (0.10s) post-processing to form final onset/offset events
- **Training data:** Gold tier (9,426 clips, verified timestamps); Silver tier
  (~10,000 clips) added in a later retrain to test whether more (noisier) data helps
  — `[TBD: result of Gold+Silver vs Gold-only]`

### 2.3 Wav2Vec2 experiment

- **Encoder:** `facebook/wav2vec2-base` (pretrained, frozen feature extractor) with a
  BiGRU head producing per-frame logits at the encoder's native ~50fps / 20ms stride
  (320-sample stride), trained with `BCEWithLogitsLoss` and `pos_weight=2.6351`
  (computed from the positive/negative frame imbalance; an earlier run without this
  fix suffered probability collapse)
- **Result (5,000-Gold-clip run):** Event-F1 0.1811, Segment Dice 0.5597, Combined
  0.7408 — below the CRNN's 0.8418, so the CRNN remains our primary submission
- **Status:** `[TBD]` — a full 9,426-Gold-clip run is planned; if it closes the gap,
  we will consider ensembling the two models' frame-level probabilities before
  decoding

### 2.4 Bronze-tier weak supervision (novelty)

The Bronze tier carries no timestamps, only a clip-level signal (presence/absence of
noise tags in the transcript) — unused by any team unless explicitly exploited, since
it cannot be trained on directly in the same way as Gold/Silver.

**Method:** self-training with a tag-agreement filter. We run our trained CRNN over
every Bronze clip to get its own frame-level prediction, then keep a clip as a
pseudo-labeled training example only when the model's prediction *agrees* with the
weak tag label:
- Tags say "noise present" **and** the model confidently predicts at least one event
  (mean probability over predicted-positive frames >= 0.6) -> accept, pseudo-label =
  the model's own predicted events
- Tags say "no noise" **and** the model predicts no events -> accept as a confident
  negative example
- Disagreement either way -> discard (too risky to trust blindly)

This is deliberately conservative: we would rather use a smaller, higher-confidence
subset of Bronze than risk training on self-generated noise. Implementation:
`src/bronze_weak_supervision.py`.

**Result:** `[TBD]` — experiment designed and implemented; full run (requires
HF-token-backed audio access and the trained checkpoint) pending as of this draft.

---

## 3. Track 2 — Noise Event Removal

### 3.1 Approach

Rather than design a denoising architecture from scratch, we build on the baseline
the organizers published (`github.com/nagshubhadip/noise-event-detection-and-removal`,
`track2_removal/versions/v-1_indoml2026_track2`): a mask-based enhancement network
conditioned on a pretrained speech encoder, trained on simulated noisy/clean mixture
pairs and fine-tuned on real validation pairs. Our contribution is (a) running and
validating this pipeline end-to-end on our own infrastructure, and (b) conditioning
it on **our own Track 1 CRNN's onset/offset predictions** rather than the baseline's
default event source, so the Track 2 submission reflects our own detection pipeline
rather than an unmodified default.

### 3.2 Architecture

- **Mask network:** operates on the STFT magnitude of the noisy mixture, conditioned
  via FiLM on an encoder embedding (SraVaani or WavLM) of the clip
- **Training:** simulated mixtures built from Gold-tier clean speech + isolated
  noise-event spans at sampled SNRs (-5 to 12 dB), then fine-tuned on real validation
  (noisy, clean) pairs at a lower learning rate, holding out 15% for honest model
  selection
- **Output stage:** blended estimate (`0.85 * estimate + 0.15 * original mixture`)
  with RMS-level restoration before the mandated ASR pass — free under SI-SDR
  (scale-invariant) but important for ASR quality
- **Mandated ASR:** `ARTPARK-IISc/SraVaani-1.0` (frozen), used both to produce
  `transcripts.jsonl` for scoring and matched by the competition's hidden anti-cheat
  audit, which re-runs the same ASR on a sample of our enhanced audio

### 3.3 Evaluation

Local scorer (`src/eval_local_removal.py`) reproduces the official metric exactly:
`Combined = SI-SDR(synthetic subset) + 100 * Delta-WER(full validation set)`, using
the organizers' own SI-SDR formula and pooled-WER computation, so our local number
should track the leaderboard number closely.

### 3.4 Results

`[TBD]` — first pipeline run in progress as of this draft (training/fine-tuning
stages completed; evaluation blocked on attaching the real validation dataset,
being resolved). No Track 2 submission made yet.

---

## 4. Data Usage

| Tier | Size | Used for |
|---|---|---|
| Gold | ~20h (9,426 clips) | Primary CRNN training; Wav2Vec2 training; Track 2 mixture simulation (clean speech + noise banks) |
| Silver | ~100h | CRNN retrain (added to Gold) — `[TBD: result]` |
| Bronze | ~30h | Weak-supervision self-training experiment (Section 2.4) — `[TBD: result]` |
| Validation (Codabench-provided) | per track | Local scoring before each submission, via `eval_local.py` (Track 1) and `eval_local_removal.py` (Track 2) |

No data outside the organizer-provided Vaani corpus and Codabench validation/test
sets was used for training or evaluation.

---

## 5. Pre-trained Models Used

| Model | Source | Role |
|---|---|---|
| `facebook/wav2vec2-base` | Hugging Face | Alternative Track 1 encoder (Section 2.3) |
| Track 2 baseline (mask net + SraVaani/WavLM FiLM encoder) | Organizer-published baseline repo | Starting point for our Track 2 model, adapted to our own Track 1 predictions |
| `ARTPARK-IISc/SraVaani-1.0` | Hugging Face | Mandated, frozen ASR for Track 2 transcript generation (unmodified, per competition rules) |

All are documented here per the competition's reproducibility requirement
("pre-trained models are permitted and must be documented").

---

## 6. Results Summary

| Track | Metric | Validation | Leaderboard |
|---|---|---|---|
| Track 1 — Detection | Combined (Event-F1 + Dice) | 0.8418 (CRNN, Gold-only) | **0.9221**, rank 44/78 (as of Sep 22 submission) |
| Track 1 — Detection | Combined, Wav2Vec2 (5k Gold) | 0.7408 | not submitted (below CRNN) |
| Track 1 — Detection | Combined, CRNN + Silver | `[TBD]` | `[TBD]` |
| Track 2 — Removal | Combined (SI-SDR + 100*ΔWER) | `[TBD]` | not yet submitted |

---

## 7. Team Contributions

- **Shubham Warkade** — CRNN baseline (features, model, training, inference),
  Gold+Silver retraining, Wav2Vec2 experiments, local Track 1 evaluator
- **Ankit Kushwaha** — Track 2 pipeline execution (Kaggle/GPU), model training and
  fine-tuning runs
- **Rithish K** — Track 2 adaptation to use our own Track 1 predictions as the
  conditioning event source
- **Kanak Baghel** — data pipeline, Bronze-tier weak supervision, local Track 2
  evaluator, submission management across both tracks, this report, repo/branch
  coordination

---

## 8. Limitations & Future Work

- Wav2Vec2 has not yet been tried at full Gold scale (9,426 clips); result pending.
- Bronze weak-supervision pseudo-labels have not yet been validated in a full
  retrain-and-compare cycle.
- Track 2 has no submission yet as of this draft; the baseline pipeline is being
  debugged (validation data attachment) rather than improved on.
- Given more time, an ensemble of CRNN + Wav2Vec2 for Track 1, and a from-scratch
  (rather than adapted-baseline) architecture for Track 2, are natural next steps.
