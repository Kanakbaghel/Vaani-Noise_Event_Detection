"""Quick download of 3 audio files to test caching"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.audio_loader import AudioLoader
import time

print("Downloading 3 audio files...")
print("This will take 2-5 minutes.\n")

loader = AudioLoader(
    cache_dir="data/cache/audio_clips",
    target_sr=16000,
    use_streaming=False
)

test_clips = ["train_000000", "train_000001", "train_000100"]

for i, clip_id in enumerate(test_clips, 1):
    print(f"\n[{i}/3] Downloading {clip_id}...")
    start = time.time()
    try:
        audio, sr = loader.load_audio(clip_id)
        elapsed = time.time() - start
        print(f"✅ Done in {elapsed:.1f}s - Shape: {audio.shape}, Duration: {len(audio)/sr:.2f}s")
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n" + "="*60)
stats = loader.get_cache_stats()
print(f"✅ Cached {stats['num_cached']} files ({stats['total_size_mb']} MB)")
print(f"Location: {stats['cache_dir']}")
print("="*60)
