"""
download_one_audio.py
Simple script to download just 1 audio file for verification
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.audio_loader import AudioLoader

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 70)
print("DOWNLOADING 1 AUDIO FILE FOR VERIFICATION")
print("=" * 70)

# Get first sample info from unified.jsonl
print("\n1. Reading first sample from unified.jsonl...")
with open("data/processed/unified.jsonl", "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

print(f"\n   clip_id: {sample['clip_id']}")
print(f"   Language: {sample['language']}")
print(f"   Duration: {sample['duration']}s")
print(f"   Tier: {sample['tier']}")
print(f"   Transcript: {sample['transcript'][:60]}...")

# Initialize loader
print("\n2. Initializing AudioLoader...")
loader = AudioLoader(
    cache_dir="data/cache/audio_clips",
    target_sr=16000,
    use_streaming=True
)
print(f"   ✓ Cache directory: {loader.cache_dir}")

# Download audio
print(f"\n3. Downloading audio for {sample['clip_id']}...")
print("   (This may take 30-60 seconds...)")

try:
    import time
    start = time.time()
    
    audio, sr = loader.load_audio(sample['clip_id'])
    
    elapsed = time.time() - start
    
    # Calculate duration
    actual_duration = len(audio) / sr
    expected_duration = sample['duration']
    diff = abs(actual_duration - expected_duration)
    
    print(f"\n✅ SUCCESS!")
    print(f"\n   Download time: {elapsed:.2f}s")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Expected duration: {expected_duration:.2f}s")
    print(f"   Actual duration: {actual_duration:.2f}s")
    print(f"   Difference: {diff:.3f}s")
    
    # Check cache file
    cache_path = loader._get_cache_path(sample['clip_id'])
    if cache_path.exists():
        file_size = cache_path.stat().st_size / 1024
        print(f"\n📁 Cache file created:")
        print(f"   Location: {cache_path}")
        print(f"   Size: {file_size:.1f} KB")
        
        # Provide full path
        import os
        full_path = os.path.abspath(cache_path)
        print(f"\n💡 Full path to audio file:")
        print(f"   {full_path}")
        print(f"\n   You can open this file in:")
        print(f"   - Windows Media Player")
        print(f"   - VLC")
        print(f"   - Audacity")
        print(f"   - Any audio player")
    
    # Verdict
    if diff < 0.1:
        print(f"\n🎉 PERFECT! Duration matches (within 0.1s tolerance)")
        print(f"   ✅ Audio loading system is WORKING CORRECTLY!")
    elif diff < 0.5:
        print(f"\n✓ Good! Duration is close (within 0.5s)")
        print(f"   ✅ Audio loading system is working")
    else:
        print(f"\n⚠️  Duration difference is {diff:.2f}s")
        print(f"   May need investigation")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nPossible issues:")
    print("  - Network timeout (HuggingFace is slow)")
    print("  - HF_TOKEN invalid")
    print("  - Internet connection issue")

print("\n" + "=" * 70)
