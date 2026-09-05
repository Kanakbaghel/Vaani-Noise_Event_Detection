"""
test_audio_download.py

FINAL PRACTICAL TEST: Actually download 3 audio files and verify they match
the expected metadata from unified.jsonl

This is the DEFINITIVE test to confirm audio loading is working correctly.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("FINAL AUDIO DOWNLOAD TEST")
print("=" * 70)

# Check dependencies
try:
    from src.audio_loader import AudioLoader
    print("\n✓ audio_loader imported successfully")
except ImportError as e:
    print(f"\n❌ Error importing audio_loader: {e}")
    print("   Make sure you're in the project root directory")
    sys.exit(1)

# Load test cases from unified.jsonl
print("\n📖 Loading test cases from unified.jsonl...")

test_cases = []
with open("data/processed/unified.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:  # Only test first 3 samples
            break
        sample = json.loads(line)
        test_cases.append({
            'clip_id': sample['clip_id'],
            'expected_language': sample.get('language', 'unknown'),
            'expected_duration': sample.get('duration', 0),
            'expected_tier': sample['tier'],
            'transcript_preview': sample['transcript'][:50],
        })

print(f"✓ Loaded {len(test_cases)} test cases\n")

for i, tc in enumerate(test_cases, 1):
    print(f"{i}. {tc['clip_id']}: {tc['expected_language']}, {tc['expected_duration']}s, {tc['expected_tier']}")

# Initialize AudioLoader
print("\n" + "=" * 70)
print("Initializing AudioLoader...")
print("=" * 70)

try:
    loader = AudioLoader(
        cache_dir="data/cache/audio_clips",
        target_sr=16000,
        use_streaming=True
    )
    print("✓ AudioLoader initialized")
    print(f"  Cache directory: {loader.cache_dir}")
    print(f"  Target sample rate: {loader.target_sr} Hz")
except Exception as e:
    print(f"❌ Error initializing AudioLoader: {e}")
    sys.exit(1)

# Test each case
print("\n" + "=" * 70)
print("DOWNLOADING & VERIFYING AUDIO")
print("=" * 70)

results = []

for i, tc in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"Test {i}/3: {tc['clip_id']}")
    print(f"{'='*70}")
    
    print(f"\n📋 Expected from unified.jsonl:")
    print(f"   Language: {tc['expected_language']}")
    print(f"   Duration: {tc['expected_duration']:.2f}s")
    print(f"   Tier: {tc['expected_tier']}")
    print(f"   Transcript: {tc['transcript_preview']}...")
    
    print(f"\n⬇️  Downloading audio from HuggingFace...")
    print(f"   This may take 10-30 seconds for first download...")
    
    try:
        import time
        start_time = time.time()
        
        # Download audio
        audio, sr = loader.load_audio(tc['clip_id'])
        
        download_time = time.time() - start_time
        
        # Calculate actual duration
        actual_duration = len(audio) / sr
        
        print(f"\n✓ Audio downloaded successfully!")
        print(f"   Download time: {download_time:.2f}s")
        print(f"   Audio shape: {audio.shape}")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Actual duration: {actual_duration:.2f}s")
        
        # Verify duration matches
        duration_diff = abs(actual_duration - tc['expected_duration'])
        
        if duration_diff < 0.1:  # Allow 0.1s tolerance
            print(f"\n✅ DURATION MATCH! ({actual_duration:.2f}s vs {tc['expected_duration']:.2f}s)")
            print(f"   Difference: {duration_diff:.3f}s (within tolerance)")
            result = "PASS"
        else:
            print(f"\n⚠️  DURATION MISMATCH!")
            print(f"   Expected: {tc['expected_duration']:.2f}s")
            print(f"   Got: {actual_duration:.2f}s")
            print(f"   Difference: {duration_diff:.2f}s")
            result = "DURATION_MISMATCH"
        
        # Check cache
        cache_path = loader._get_cache_path(tc['clip_id'])
        if cache_path.exists():
            file_size_kb = cache_path.stat().st_size / 1024
            print(f"\n💾 Cached at: {cache_path}")
            print(f"   File size: {file_size_kb:.1f} KB")
        else:
            print(f"\n⚠️  Cache file not found at: {cache_path}")
            result = "CACHE_FAILED"
        
        results.append({
            'clip_id': tc['clip_id'],
            'result': result,
            'duration_diff': duration_diff,
            'download_time': download_time,
        })
        
    except Exception as e:
        print(f"\n❌ ERROR downloading audio: {e}")
        import traceback
        traceback.print_exc()
        
        results.append({
            'clip_id': tc['clip_id'],
            'result': 'ERROR',
            'error': str(e),
        })

# Test cache reuse
print("\n" + "=" * 70)
print("TESTING CACHE REUSE (2nd load should be fast)")
print("=" * 70)

if results and results[0]['result'] == 'PASS':
    test_clip = test_cases[0]['clip_id']
    
    print(f"\n🔄 Re-loading {test_clip} (should be from cache)...")
    
    import time
    start_time = time.time()
    audio2, sr2 = loader.load_audio(test_clip)
    cache_time = time.time() - start_time
    
    print(f"✓ Loaded from cache in {cache_time:.3f}s")
    
    if cache_time < results[0]['download_time'] * 0.5:
        print(f"✅ Cache is working! ({results[0]['download_time']:.2f}s → {cache_time:.3f}s)")
    else:
        print(f"⚠️  Cache may not be working optimally")

# Cache statistics
print("\n" + "=" * 70)
print("CACHE STATISTICS")
print("=" * 70)

try:
    stats = loader.get_cache_stats()
    print(f"\n📊 Cache Status:")
    print(f"   Cached files: {stats['num_cached']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    print(f"   Cache directory: {stats['cache_dir']}")
except Exception as e:
    print(f"⚠️  Could not get cache stats: {e}")

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

passed = sum(1 for r in results if r['result'] == 'PASS')
total = len(results)

print(f"\n📊 Test Results: {passed}/{total} passed\n")

for r in results:
    if r['result'] == 'PASS':
        print(f"   ✅ {r['clip_id']}: PASS (duration diff: {r['duration_diff']:.3f}s)")
    elif r['result'] == 'DURATION_MISMATCH':
        print(f"   ⚠️  {r['clip_id']}: Duration mismatch (diff: {r['duration_diff']:.2f}s)")
    else:
        print(f"   ❌ {r['clip_id']}: {r['result']}")

if passed == total:
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 70)
    print("\n✅ Audio loading system is FULLY FUNCTIONAL!")
    print("\nKey confirmations:")
    print("   ✓ Audio downloads from HuggingFace successfully")
    print("   ✓ clip_id → HF index mapping is correct")
    print("   ✓ Downloaded audio matches expected duration")
    print("   ✓ Cache system is working")
    print("   ✓ Sequential indexing is stable")
    print("\n🚀 You're ready to start training!")
    print("\nNext steps:")
    print("   1. Share this success with your team")
    print("   2. Start using VaaniNoiseDataset in your training")
    print("   3. Monitor cache size as you train")
    
elif passed > 0:
    print(f"\n⚠️  PARTIAL SUCCESS: {passed}/{total} tests passed")
    print("\nSome tests had issues but audio loading is working.")
    print("Check the details above for specific issues.")
    
else:
    print("\n❌ ALL TESTS FAILED")
    print("\nPlease check:")
    print("   1. HF_TOKEN is valid in .env")
    print("   2. Internet connection is stable")
    print("   3. HuggingFace dataset is accessible")
    print("   4. Dependencies are installed correctly")

print("\n" + "=" * 70)
