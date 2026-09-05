"""
test_skip_optimization.py

Quick test to verify .skip() optimization works for high-index samples.
Tests that we can load train_010000 without iterating through 10k samples.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_loader import AudioLoader

print("=" * 70)
print("SKIP OPTIMIZATION TEST - High Index Sample")
print("=" * 70)

print("\nThis test verifies that loading train_010000 uses .skip()")
print("and does NOT iterate through 10,000 samples.\n")

# Initialize loader
print("Initializing AudioLoader...")
loader = AudioLoader(
    cache_dir="data/cache/audio_clips",
    target_sr=16000,
    use_streaming=True
)
print("✓ Loader initialized\n")

# Test with a high-index sample
test_clip = "train_010000"
print(f"Loading {test_clip} (index 10,000)...")
print("If optimization works: should complete in ~30-60 seconds")
print("If NO optimization: would take 5-10 minutes!\n")

start_time = time.time()

try:
    audio, sr = loader.load_audio(test_clip)
    load_time = time.time() - start_time
    
    print(f"\n✅ SUCCESS!")
    print(f"   Loaded in: {load_time:.1f} seconds")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Sample rate: {sr}")
    print(f"   Duration: {len(audio)/sr:.2f}s")
    
    if load_time < 120:
        print(f"\n🚀 OPTIMIZATION CONFIRMED!")
        print(f"   Load time ({load_time:.1f}s) indicates .skip() is working!")
    else:
        print(f"\n⚠️  SLOW LOAD")
        print(f"   Load time ({load_time:.1f}s) suggests iteration, not skip")
        
except Exception as e:
    load_time = time.time() - start_time
    print(f"\n❌ FAILED after {load_time:.1f}s")
    print(f"   Error: {e}")

print("\n" + "=" * 70)
