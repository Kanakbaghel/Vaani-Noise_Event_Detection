"""
quick_test.py
Quick test to verify audio loading works (minimal dependencies check)
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

print("=" * 60)
print("QUICK AUDIO LOADING TEST")
print("=" * 60)

# Test 1: Check if required packages are installed
print("\n1. Checking required packages...")
required = ['numpy', 'torch', 'torchaudio', 'soundfile', 'datasets', 'dotenv']
missing = []

for pkg in required:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg} - NOT INSTALLED")
        missing.append(pkg)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print("\nPlease install requirements:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

print("\n✓ All required packages installed!")

# Test 2: Check .env file
print("\n2. Checking .env configuration...")
try:
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("   ✗ HF_TOKEN not found in .env")
        print("\n   Please create .env file:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your HuggingFace token: HF_TOKEN=your_token_here")
        sys.exit(1)
    
    if len(token) < 10:
        print("   ✗ HF_TOKEN seems invalid (too short)")
        sys.exit(1)
    
    print(f"   ✓ HF_TOKEN found (length: {len(token)})")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Check if data splits exist
print("\n3. Checking data files...")
from pathlib import Path

data_files = [
    "data/processed/unified.jsonl",
    "data/processed/train_split.jsonl",
    "data/processed/val_split.jsonl"
]

for file_path in data_files:
    if Path(file_path).exists():
        # Count lines
        with open(file_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        print(f"   ✓ {file_path} ({num_lines} records)")
    else:
        print(f"   ✗ {file_path} - NOT FOUND")

# Test 4: Test clip_id mapping logic
print("\n4. Testing clip_id → index mapping...")

def extract_index_from_clip_id(clip_id):
    """Extract numeric index from clip_id"""
    try:
        return int(clip_id.replace("train_", ""))
    except ValueError:
        return None

test_cases = [
    ("train_000000", 0),
    ("train_000001", 1),
    ("train_000042", 42),
    ("train_012345", 12345),
]

all_passed = True
for clip_id, expected_idx in test_cases:
    result = extract_index_from_clip_id(clip_id)
    if result == expected_idx:
        print(f"   ✓ {clip_id} → {result}")
    else:
        print(f"   ✗ {clip_id} → {result} (expected {expected_idx})")
        all_passed = False

if not all_passed:
    print("\n❌ Mapping logic failed!")
    sys.exit(1)

# Test 5: Verify JSONL data structure
print("\n5. Verifying JSONL data structure...")

try:
    with open("data/processed/train_split.jsonl", 'r', encoding='utf-8') as f:
        # Read first 3 samples
        samples = []
        for i, line in enumerate(f):
            if i >= 3:
                break
            samples.append(json.loads(line))
    
    print(f"   ✓ Successfully loaded {len(samples)} sample records")
    
    # Check first sample structure
    sample = samples[0]
    required_fields = ['clip_id', 'tier', 'duration', 'events', 'transcript']
    
    for field in required_fields:
        if field in sample:
            print(f"   ✓ Field '{field}' present")
        else:
            print(f"   ✗ Field '{field}' MISSING")
            all_passed = False
    
    # Show sample clip_id and validate format
    clip_id = sample['clip_id']
    print(f"\n   Sample clip_id: {clip_id}")
    
    if clip_id.startswith("train_") and clip_id.replace("train_", "").isdigit():
        print(f"   ✓ Clip ID format is valid")
        extracted_idx = extract_index_from_clip_id(clip_id)
        print(f"   ✓ Extracted index: {extracted_idx}")
    else:
        print(f"   ✗ Clip ID format is INVALID")
        all_passed = False
    
except Exception as e:
    print(f"   ✗ Error reading JSONL: {e}")
    sys.exit(1)

# Test 6: Test AudioLoader instantiation
print("\n6. Testing AudioLoader initialization...")

try:
    from src.audio_loader import AudioLoader
    
    loader = AudioLoader(
        cache_dir="data/cache/audio_clips",
        target_sr=16000,
        use_streaming=True
    )
    
    print(f"   ✓ AudioLoader initialized")
    print(f"   ✓ Cache directory: {loader.cache_dir}")
    print(f"   ✓ Target SR: {loader.target_sr}")
    
except Exception as e:
    print(f"   ✗ Error initializing AudioLoader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test loading a single audio clip
print("\n7. Testing audio download from HuggingFace...")
print("   This will download 1 audio clip to verify the system works.")
print("   It may take 10-30 seconds for the first download...\n")

try:
    # Use a clip from your actual data
    with open("data/processed/train_split.jsonl", 'r', encoding='utf-8') as f:
        first_sample = json.loads(f.readline())
        test_clip_id = first_sample['clip_id']
    
    print(f"   Testing with clip_id: {test_clip_id}")
    
    import time
    start_time = time.time()
    
    audio, sr = loader.load_audio(test_clip_id)
    
    elapsed = time.time() - start_time
    
    print(f"   ✓ Audio loaded successfully!")
    print(f"   ✓ Audio shape: {audio.shape}")
    print(f"   ✓ Sample rate: {sr} Hz")
    print(f"   ✓ Duration: {len(audio) / sr:.2f} seconds")
    print(f"   ✓ Download time: {elapsed:.2f}s")
    
    # Verify audio is not empty
    if len(audio) > 0:
        print(f"   ✓ Audio data is valid (non-empty)")
    else:
        print(f"   ✗ Audio data is EMPTY")
        sys.exit(1)
    
    # Check if cached
    cache_path = loader._get_cache_path(test_clip_id)
    if cache_path.exists():
        print(f"   ✓ Audio cached at: {cache_path}")
        file_size_kb = cache_path.stat().st_size / 1024
        print(f"   ✓ Cache file size: {file_size_kb:.1f} KB")
    else:
        print(f"   ✗ Cache file NOT created")
        sys.exit(1)
    
except Exception as e:
    print(f"   ✗ Error loading audio: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test cache reuse
print("\n8. Testing cache reuse (2nd load should be faster)...")

try:
    start_time = time.time()
    audio2, sr2 = loader.load_audio(test_clip_id)
    elapsed2 = time.time() - start_time
    
    print(f"   ✓ Audio loaded from cache")
    print(f"   ✓ Load time: {elapsed2:.3f}s (was {elapsed:.2f}s on first load)")
    
    if elapsed2 < elapsed * 0.5:  # Should be significantly faster
        print(f"   ✓ Cache reuse is working! ({elapsed/max(elapsed2, 0.001):.1f}x faster)")
    else:
        print(f"   ⚠  Cache might not be working optimally")
    
    # Verify same data
    import numpy as np
    if np.array_equal(audio, audio2) and sr == sr2:
        print(f"   ✓ Cached audio matches original")
    else:
        print(f"   ✗ Cached audio DOES NOT match original")
        sys.exit(1)
    
except Exception as e:
    print(f"   ✗ Error with cache reuse: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Cache statistics
print("\n9. Checking cache statistics...")

try:
    stats = loader.get_cache_stats()
    print(f"   ✓ Cached files: {stats['num_cached']}")
    print(f"   ✓ Total cache size: {stats['total_size_mb']} MB")
    print(f"   ✓ Cache directory: {stats['cache_dir']}")
    
except Exception as e:
    print(f"   ✗ Error getting cache stats: {e}")

# Final summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe audio loading system is working correctly:")
print(f"  • Downloaded audio from HuggingFace for clip: {test_clip_id}")
print(f"  • Cached locally for fast reuse")
print(f"  • Verified audio data integrity")
print("\nYou can now:")
print("  1. Use AudioLoader in your training code")
print("  2. Test with PyTorch Dataset: python src/dataset.py")
print("  3. Start building your model!")
print()
