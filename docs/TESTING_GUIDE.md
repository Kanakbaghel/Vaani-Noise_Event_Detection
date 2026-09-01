# Testing Guide - Audio Loading System

## ✅ What We've Verified So Far

The **setup verification** script has already confirmed:

1. ✅ All required files exist (`audio_loader.py`, `dataset.py`, data splits)
2. ✅ `.env` file is configured with HF_TOKEN
3. ✅ Data structure is correct (clip_id format, JSONL structure)
4. ✅ Audio loader code structure is complete
5. ✅ Clip ID → Index mapping logic is correct
6. ✅ Cache directory is gitignored

**Run verification:** `python verify_setup.py` ✅ **PASSED**

---

## 🔧 Installation Steps

Before testing actual audio download, you need to install dependencies:

###Option 1: Install All Requirements (Recommended)
```bash
pip install -r requirements.txt
```

**Note:** This will download ~2-3 GB of packages (PyTorch, torchaudio, etc.). It may take 15-30 minutes depending on your internet speed.

### Option 2: Install Only Essential Packages (For Quick Testing)
```bash
pip install numpy datasets soundfile python-dotenv
```

Then later install torch/torchaudio when needed:
```bash
pip install torch torchaudio
```

---

## 🧪 Testing Actual Audio Download

Once packages are installed, test if audio is downloading correctly:

### Test 1: Quick Audio Test
```bash
python quick_test.py
```

This will:
- ✅ Check all required packages are installed  
- ✅ Verify .env configuration
- ✅ **Download 1 audio clip from HuggingFace** based on clip_id
- ✅ Save it to `data/cache/audio_clips/`
- ✅ Verify the audio data is valid
- ✅ Test cache reuse (2nd load should be instant)

**Expected output:**
```
7. Testing audio download from HuggingFace...
   Testing with clip_id: train_020733
   Fetching train_020733 from HuggingFace...
   ✓ Audio loaded successfully!
   ✓ Audio shape: (73696,)
   ✓ Sample rate: 16000 Hz
   ✓ Duration: 4.61 seconds
   ✓ Download time: 12.34s
   ✓ Audio cached at: data\cache\audio_clips\train_020733.wav
   ✓ Cache file size: 287.4 KB
```

### Test 2: Full System Test
```bash
python test_audio_system.py
```

This runs 3 comprehensive tests:
1. AudioLoader functionality
2. PyTorch Dataset integration
3. Cache reuse verification

---

## 🔍 Manual Verification

### Verify Audio Was Downloaded

1. **Check cache directory:**
   ```bash
   dir data\cache\audio_clips
   ```
   
   You should see `.wav` files like:
   ```
   train_020733.wav
   train_000001.wav
   ...
   ```

2. **Play the audio file** (optional):
   - Navigate to `data/cache/audio_clips/`
   - Open any `.wav` file with your media player
   - You should hear actual speech audio from the Vaani dataset

3. **Check file size**:
   - Each audio file should be ~100-500 KB
   - If file size is 0 or very small (< 10 KB), something went wrong

### Verify Clip ID Mapping

The audio loader maps `clip_id` → HF dataset index:

- `train_000000` → HF index 0
- `train_020733` → HF index 20733
- `train_012345` → HF index 12345

To verify this is working:

```python
python
>>> from src.audio_loader import AudioLoader
>>> loader = AudioLoader()
>>> audio, sr = loader.load_audio("train_000001")
>>> print(f"Loaded audio: shape={audio.shape}, sr={sr}")
```

Expected: Audio shape should be (N,) where N > 0, sr should be 16000

---

## 📊 What Gets Downloaded?

When you run the tests, here's what happens:

1. **First time loading** `train_020733`:
   ```
   clip_id: train_020733
   └─> Extract index: 20733
       └─> Stream HF dataset to index 20733
           └─> Download audio
               └─> Resample to 16kHz
                   └─> Save to: data/cache/audio_clips/train_020733.wav
                       └─> Return audio array
   ```

2. **Second time loading** `train_020733`:
   ```
   clip_id: train_020733
   └─> Check cache: data/cache/audio_clips/train_020733.wav
       └─> File exists! Load directly (instant)
           └─> Return audio array
   ```

---

## ✅ Success Criteria

After running tests, you should see:

### 1. Cache Directory Created
```
data/
└── cache/
    └── audio_clips/
        ├── train_020733.wav  (✅ exists)
        ├── train_000001.wav  (✅ exists)
        └── ...
```

### 2. Audio Files Are Valid
- File size: ~100-500 KB per clip
- Format: WAV, 16kHz mono
- Duration: matches the `duration` field in JSONL

### 3. Mapping Works Correctly
From your `train_split.jsonl`:
```json
{"clip_id": "train_020733", "language": "Garo", "duration": 4.606, ...}
```

Downloaded audio should:
- ✅ Be from index 20733 in HF dataset
- ✅ Be in Garo language
- ✅ Be ~4.6 seconds long (at 16kHz → ~73,696 samples)

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors
**Solution:** Install missing packages:
```bash
pip install numpy datasets soundfile python-dotenv torch torchaudio
```

### Issue: "HF_TOKEN not found"
**Solution:** Check your `.env` file:
```bash
type .env
```
Should show: `HF_TOKEN=hf_...`

### Issue: Download is very slow
**Expected:** First download takes 10-30 seconds per clip (streaming from HF)  
**Workaround:** Once cached, subsequent loads are instant

### Issue: Audio file is 0 bytes
**Possible causes:**
- HF token is invalid
- Network connection issue
- Dataset access denied

**Solution:** 
1. Check HF token is valid: https://huggingface.co/settings/tokens
2. Verify dataset access: https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset

### Issue: "Could not find clip_id in dataset"
**Cause:** The clip_id index might exceed the dataset size
**Solution:** Use clip_ids from your actual JSONL files (they're verified to exist)

---

## 🎯 Next Steps After Successful Testing

Once tests pass:

1. **Share with team:**
   - Everyone runs `python verify_setup.py`
   - Everyone runs `python quick_test.py`
   - Confirm audio files appear in cache

2. **Start using in training:**
   ```python
   from src.dataset import VaaniNoiseDataset
   from torch.utils.data import DataLoader
   
   dataset = VaaniNoiseDataset("data/processed/train_split.jsonl")
   loader = DataLoader(dataset, batch_size=16)
   
   for batch in loader:
       audio = batch['audio']  # (16, max_length)
       events = batch['events']
       # Your training code here
   ```

3. **Monitor cache growth:**
   ```python
   from src.audio_loader import AudioLoader
   loader = AudioLoader()
   stats = loader.get_cache_stats()
   print(f"Cached: {stats['num_cached']} files, {stats['total_size_mb']} MB")
   ```

---

## 📋 Testing Checklist

- [ ] `verify_setup.py` passes (all ✓)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `quick_test.py` downloads audio successfully
- [ ] Cache directory contains `.wav` files
- [ ] Audio files have valid size (~100-500 KB)
- [ ] Second load is significantly faster (cache working)
- [ ] Can play audio files and hear speech
- [ ] Clip ID mapping is correct (matches JSONL duration)

---

**Once all checkboxes are ✓, your audio loading system is fully functional!** 🎉
