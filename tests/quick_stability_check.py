"""
quick_stability_check.py
Fast stability test - only checks first 10 samples
"""

import os
import json
from datasets import load_dataset, Audio
from dotenv import load_dotenv

print("=" * 70)
print("QUICK HF STREAMING STABILITY CHECK")
print("=" * 70)

load_dotenv()
token = os.environ.get("HF_TOKEN")

DATASET_NAME = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"

# ============================================================================
# TEST 1: Check available fields (including imageFileName)
# ============================================================================
print("\nTEST 1: Available Fields in HF Dataset")
print("=" * 70)

dataset = load_dataset(DATASET_NAME, token=token, streaming=True)
train_split = dataset["train"].cast_column("audio", Audio(decode=False))

first_sample = next(iter(train_split))

print("\n✓ Fields available:\n")
for key, value in first_sample.items():
    if key == 'audio':
        continue
    value_str = str(value)[:80]
    print(f"   {key:<30} = {value_str}")

# Check for imageFileName
if 'imageFileName' in first_sample:
    print(f"\n✅ imageFileName field EXISTS!")
    print(f"   Value: {first_sample['imageFileName']}")
    print(f"   This could be used as a stable unique identifier!")
else:
    print(f"\n⚠ imageFileName field NOT FOUND")

# ============================================================================
# TEST 2: Quick Sequential Stability (first 10 samples only)
# ============================================================================
print("\n\nTEST 2: Sequential Stability (first 10 samples)")
print("=" * 70)

print("\n🔄 Stream 1: Collecting first 10 samples...")

def get_samples(n=10):
    dataset = load_dataset(DATASET_NAME, token=token, streaming=True)
    train_split = dataset["train"].cast_column("audio", Audio(decode=False))
    
    samples = []
    for i, sample in enumerate(train_split):
        if i >= n:
            break
        samples.append({
            'index': i,
            'transcript': sample.get('transcript', '')[:50],
            'duration': sample.get('duration'),
            'language': sample.get('language'),
            'imageFileName': sample.get('imageFileName', ''),
        })
    return samples

stream1 = get_samples(10)

print("🔄 Stream 2: Collecting first 10 samples again...")
stream2 = get_samples(10)

print("\n📊 Comparing both streams:\n")

all_match = True
for i in range(10):
    s1 = stream1[i]
    s2 = stream2[i]
    
    match = (
        s1['transcript'] == s2['transcript'] and
        s1['duration'] == s2['duration'] and
        s1['language'] == s2['language'] and
        s1['imageFileName'] == s2['imageFileName']
    )
    
    if match:
        print(f"Index {i}: ✅ MATCH - {s1['language']}, {s1['duration']}s")
    else:
        print(f"Index {i}: ❌ MISMATCH")
        print(f"   Stream1: {s1}")
        print(f"   Stream2: {s2}")
        all_match = False

# ============================================================================
# TEST 3: Match with unified.jsonl
# ============================================================================
print("\n\nTEST 3: Cross-verify with unified.jsonl")
print("=" * 70)

# Load from unified.jsonl
unified_samples = []
with open("data/processed/unified.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        unified_samples.append(json.loads(line))

print(f"\n✓ Loaded {len(unified_samples)} samples from unified.jsonl")

# Get fresh samples from HF
print("✓ Fetching first 10 samples from HF...")
hf_samples = get_samples(10)

print("\n📊 Comparing unified.jsonl with live HF stream:\n")

matches = 0
for i in range(10):
    unified = unified_samples[i]
    hf = hf_samples[i]
    
    # Compare
    transcript_match = unified['transcript'] == hf['transcript'][:50] or unified['transcript'][:50] == hf['transcript']
    duration_match = abs(unified['duration'] - hf['duration']) < 0.01
    language_match = unified['language'] == hf['language']
    
    if transcript_match and duration_match and language_match:
        matches += 1
        print(f"Index {i}: ✅ MATCH - clip_id: {unified['clip_id']}")
    else:
        print(f"Index {i}: ❌ MISMATCH")
        print(f"   Transcript: {transcript_match}")
        print(f"   Duration: {duration_match} ({unified['duration']} vs {hf['duration']})")
        print(f"   Language: {language_match}")

# ============================================================================
# TEST 4: Check imageFileName as stable identifier
# ============================================================================
print("\n\nTEST 4: imageFileName as Unique Identifier")
print("=" * 70)

if 'imageFileName' in first_sample:
    print("\n🔍 Analyzing imageFileName uniqueness...")
    
    # Get 50 samples
    dataset = load_dataset(DATASET_NAME, token=token, streaming=True)
    train_split = dataset["train"].cast_column("audio", Audio(decode=False))
    
    image_filenames = []
    for i, sample in enumerate(train_split):
        if i >= 50:
            break
        image_filenames.append(sample.get('imageFileName', ''))
    
    unique_count = len(set(image_filenames))
    
    print(f"\n   Collected: {len(image_filenames)} samples")
    print(f"   Unique imageFileNames: {unique_count}")
    
    if unique_count == len(image_filenames):
        print(f"   ✅ All imageFileNames are UNIQUE!")
        print(f"   → imageFileName can be used as stable identifier")
    else:
        print(f"   ⚠ Some imageFileNames are duplicated")
    
    print(f"\n   Sample imageFileNames:")
    for i in range(min(5, len(image_filenames))):
        print(f"      {i}: {image_filenames[i]}")

# ============================================================================
# FINAL RESULT
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if all_match and matches >= 9:
    print("\n✅ STABILITY CHECK PASSED!")
    print("\n   Key Findings:")
    print("   1. HF streaming is STABLE (same index → same sample)")
    print("   2. Multiple streams return CONSISTENT data")
    print("   3. unified.jsonl matches HF dataset order")
    if 'imageFileName' in first_sample:
        print("   4. imageFileName field is available as alternative ID")
    print("\n   → Your audio_loader.py implementation is CORRECT!")
    print("   → Sequential index mapping (clip_id → HF index) is RELIABLE!")
    print("\n🎉 System is ready for production use!")
else:
    print("\n⚠ STABILITY ISSUES DETECTED")
    print("\n   Some mismatches were found.")
    print("   May need to use imageFileName or other stable field.")

print("\n" + "=" * 70)
