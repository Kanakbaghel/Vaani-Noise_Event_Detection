"""
Simple test to verify audio_loader.py caching works
Uses mock/local audio to test cache functionality WITHOUT downloading from HF
"""
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Create a test audio file manually
print("=" * 70)
print("TESTING AUDIO_LOADER CACHING (LOCAL TEST)")
print("=" * 70)

cache_dir = Path("data/cache/audio_clips")
cache_dir.mkdir(parents=True, exist_ok=True)

# Create a fake audio file to test cache
test_clip_id = "train_999999"
test_cache_path = cache_dir / f"{test_clip_id}.wav"

print(f"\n1. Creating test audio file: {test_cache_path}")

# Generate fake audio data
import soundfile as sf
fake_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
sr = 16000

# Save it
sf.write(test_cache_path, fake_audio, sr)
print(f"   ✓ Created {test_cache_path.stat().st_size / 1024:.2f} KB")

# Now test if AudioLoader loads from cache
print(f"\n2. Testing AudioLoader cache loading...")

from src.audio_loader import AudioLoader

loader = AudioLoader(cache_dir=str(cache_dir), target_sr=16000)

# Test 1: Load from cache
print(f"\n   Test 1: Loading {test_clip_id} (should load from cache)")
start = time.time()
try:
    audio, loaded_sr = loader.load_audio(test_clip_id)
    elapsed = time.time() - start
    
    print(f"   ✅ Loaded in {elapsed:.3f}s")
    print(f"   Shape: {audio.shape}, SR: {loaded_sr}")
    
    if elapsed < 0.1:
        print(f"   ✅ FAST LOAD - Cache is working! (<0.1s)")
    else:
        print(f"   ⚠️  Slow load - might not be using cache")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Check cache stats
print(f"\n3. Checking cache statistics...")
stats = loader.get_cache_stats()
print(f"   Cached files: {stats['num_cached']}")
print(f"   Total size: {stats['total_size_mb']} MB")
print(f"   Cache directory: {stats['cache_dir']}")

# Test 3: Load again (should be instant from cache)
print(f"\n4. Loading same file again (should be instant)...")
start = time.time()
audio2, sr2 = loader.load_audio(test_clip_id)
elapsed2 = time.time() - start

print(f"   ✅ Loaded in {elapsed2:.3f}s")

if elapsed2 < 0.05:
    print(f"   ✅ CACHE WORKING! Second load was instant")
else:
    print(f"   ⚠️  Still slow - cache might have issues")

# Test 4: Verify it's the same audio
if np.allclose(audio, audio2):
    print(f"   ✅ Same audio data - cache is consistent")
else:
    print(f"   ❌ Different audio data - cache problem!")

# Cleanup
print(f"\n5. Cleaning up test file...")
test_cache_path.unlink()
print(f"   ✓ Removed {test_cache_path}")

print("\n" + "=" * 70)
print("CACHE TEST COMPLETE")
print("=" * 70)

# Check for any real cached files
real_cached = list(cache_dir.glob("train_*.wav"))
print(f"\nReal cached audio files: {len(real_cached)}")
if real_cached:
    print("Files:")
    for f in real_cached[:5]:  # Show first 5
        print(f"  - {f.name} ({f.stat().st_size / 1024:.2f} KB)")
else:
    print("(None yet - run download_10_files.py to populate cache)")

print()
