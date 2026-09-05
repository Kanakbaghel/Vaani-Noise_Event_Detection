"""
Download and test 10 audio files using non-streaming mode
This bypasses the torchcodec issue completely!
"""
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))
from src.audio_loader import AudioLoader

print("=" * 70)
print("DOWNLOADING 10 AUDIO FILES - NON-STREAMING MODE")
print("=" * 70)
print("\nThis will download files using librosa/soundfile")
print("(bypasses torchcodec/FFmpeg issue)")
print("\nThis may take 5-15 minutes for first download...")
print("Please be patient!\n")

# Get test clip IDs from JSONL
print("Reading clip IDs from unified.jsonl...")
test_indices = [0, 1, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000]
test_cases = []

with open("data/processed/unified.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i in test_indices:
            record = json.loads(line)
            test_cases.append({
                "clip_id": record["clip_id"],
                "expected_duration": record["duration"],
                "language": record.get("language", "unknown"),
                "tier": record["tier"],
                "index": i
            })
        if i > max(test_indices):
            break

print(f"✓ Found {len(test_cases)} test clips\n")
print("Test clips:")
for tc in test_cases:
    print(f"  {tc['clip_id']} (index {tc['index']:>5}): {tc['expected_duration']:.2f}s, {tc['language']}, {tc['tier']}")

# Initialize loader with NON-STREAMING mode
print("\n" + "=" * 70)
print("Initializing AudioLoader (NON-STREAMING MODE)...")
print("=" * 70)

loader = AudioLoader(
    cache_dir="data/cache/audio_clips",
    target_sr=16000,
    use_streaming=False  # ← This bypasses torchcodec!
)

print("✓ Loader initialized\n")

# Download each file
print("=" * 70)
print("DOWNLOADING FILES...")
print("=" * 70)

results = []
total_start = time.time()

for i, test in enumerate(test_cases, 1):
    clip_id = test["clip_id"]
    expected_dur = test["expected_duration"]
    
    print(f"\n[{i}/{len(test_cases)}] Loading {clip_id} (index {test['index']})...")
    
    start = time.time()
    try:
        audio, sr = loader.load_audio(clip_id)
        elapsed = time.time() - start
        
        actual_dur = len(audio) / sr
        duration_diff = abs(actual_dur - expected_dur)
        
        results.append({
            "clip_id": clip_id,
            "success": True,
            "time": elapsed,
            "expected_dur": expected_dur,
            "actual_dur": actual_dur,
            "diff": duration_diff
        })
        
        print(f"   ✅ SUCCESS in {elapsed:.1f}s")
        print(f"   Duration: {actual_dur:.2f}s (expected: {expected_dur:.2f}s, diff: {duration_diff:.2f}s)")
        print(f"   Shape: {audio.shape}, SR: {sr}")
        
    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "clip_id": clip_id,
            "success": False,
            "time": elapsed,
            "error": str(e)
        })
        print(f"   ❌ FAILED in {elapsed:.1f}s")
        print(f"   Error: {e}")

total_elapsed = time.time() - total_start

# Summary
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)

successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
print(f"Successful: {len(successful)}/{len(results)}")
print(f"Failed: {len(failed)}/{len(results)}")

if successful:
    print("\n✅ Successfully downloaded:")
    for r in successful:
        print(f"   {r['clip_id']}: {r['time']:.1f}s, duration: {r['actual_dur']:.2f}s")

if failed:
    print("\n❌ Failed downloads:")
    for r in failed:
        print(f"   {r['clip_id']}: {r['error']}")

# Cache stats
print("\n" + "=" * 70)
print("CACHE STATUS")
print("=" * 70)
stats = loader.get_cache_stats()
print(f"Cached files: {stats['num_cached']}")
print(f"Total size: {stats['total_size_mb']} MB")
print(f"Cache directory: {stats['cache_dir']}")

# Optimization verification
if len(successful) >= 2:
    print("\n" + "=" * 70)
    print("OPTIMIZATION VERIFICATION")
    print("=" * 70)
    print("Testing if high-index files loaded without iterating...")
    
    high_index_files = [r for r in successful if int(r['clip_id'].replace('train_', '')) > 1000]
    if high_index_files:
        print(f"\n✅ Successfully loaded {len(high_index_files)} high-index files!")
        print("This proves the .skip() optimization is working.")
        print("\nHigh-index files:")
        for r in high_index_files:
            idx = int(r['clip_id'].replace('train_', ''))
            print(f"   {r['clip_id']} (index {idx:>5}): {r['time']:.1f}s")

print("\n" + "=" * 70)
if len(successful) == len(results):
    print("✅ ALL FILES DOWNLOADED SUCCESSFULLY!")
    print("\nNext load will be instant from cache! 🚀")
elif len(successful) > 0:
    print(f"⚠️  {len(successful)}/{len(results)} files downloaded")
else:
    print("❌ All downloads failed")
print("=" * 70)
print()
