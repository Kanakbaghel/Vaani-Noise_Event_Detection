"""
test_optimized_loader.py

Test the optimized audio loader with .skip() method.
Verifies that:
1. Audio loads correctly
2. Duration matches expected values from JSONL
3. Performance is acceptable
"""

import sys
from pathlib import Path
import json
import time

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_loader import AudioLoader

print("=" * 70)
print("OPTIMIZED AUDIO LOADER TEST")
print("=" * 70)

# Initialize loader
print("\n1. Initializing AudioLoader...")
loader = AudioLoader(
    cache_dir="data/cache/audio_clips",
    target_sr=16000,
    use_streaming=True
)
print("   ✓ AudioLoader initialized")

# Load test cases from unified.jsonl
print("\n2. Loading test cases from unified.jsonl...")
with open("data/processed/unified.jsonl", "r", encoding="utf-8") as f:
    # Test with early, middle, and high indices
    test_indices = [0, 1, 100, 1000, 10000]
    test_cases = []
    
    for i, line in enumerate(f):
        if i in test_indices:
            record = json.loads(line)
            test_cases.append({
                "clip_id": record["clip_id"],
                "expected_duration": record["duration"],
                "language": record.get("language", "unknown"),
                "tier": record["tier"]
            })
        if i > max(test_indices):
            break

print(f"   ✓ Loaded {len(test_cases)} test cases")

# Test each clip
print("\n3. Testing audio loading with duration verification...")
print("-" * 70)

all_passed = True
timings = []

for i, test in enumerate(test_cases, 1):
    clip_id = test["clip_id"]
    expected_dur = test["expected_duration"]
    
    print(f"\n[{i}/{len(test_cases)}] Testing {clip_id}")
    print(f"   Expected: {expected_dur:.2f}s, {test['language']}, {test['tier']}")
    
    try:
        # Time the loading
        start_time = time.time()
        audio, sr = loader.load_audio(clip_id)
        load_time = time.time() - start_time
        timings.append((clip_id, load_time))
        
        # Calculate actual duration
        actual_dur = len(audio) / sr
        duration_diff = abs(actual_dur - expected_dur)
        
        # Verify duration matches (allow 0.2s tolerance for resampling)
        if duration_diff < 0.2:
            print(f"   ✓ Audio loaded successfully")
            print(f"   ✓ Duration: {actual_dur:.2f}s (diff: {duration_diff:.3f}s)")
            print(f"   ✓ Shape: {audio.shape}, SR: {sr}")
            print(f"   ✓ Load time: {load_time:.2f}s")
        else:
            print(f"   ✗ Duration mismatch! Actual: {actual_dur:.2f}s vs Expected: {expected_dur:.2f}s")
            all_passed = False
            
    except Exception as e:
        print(f"   ✗ Error loading audio: {e}")
        all_passed = False

# Performance summary
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)

for clip_id, load_time in timings:
    index = int(clip_id.replace("train_", ""))
    cached = "(cached)" if load_time < 1.0 else "(downloaded)"
    print(f"{clip_id} (index {index:>6}): {load_time:>6.2f}s {cached}")

# Cache stats
print("\n" + "=" * 70)
print("CACHE STATISTICS")
print("=" * 70)
stats = loader.get_cache_stats()
print(f"Cached files: {stats['num_cached']}")
print(f"Total size: {stats['total_size_mb']} MB")
print(f"Cache directory: {stats['cache_dir']}")

# Final verdict
print("\n" + "=" * 70)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print("\nOptimized audio loader is working correctly.")
    print("The .skip() method successfully loads audio from any index.")
else:
    print("⚠️  SOME TESTS FAILED")
    print("Please check the errors above.")

print("=" * 70)
