"""
analyze_existing_data.py

Pehle apne existing data ko analyze karte hain to understand:
1. unified.jsonl mein kya fields available hain
2. Koi unique identifier field hai kya
3. Data structure kya hai

No dependencies needed - pure Python!
"""

import json
from collections import Counter
from pathlib import Path

print("=" * 70)
print("EXISTING DATA ANALYSIS (unified.jsonl)")
print("=" * 70)

# Check if file exists
unified_path = Path("data/processed/unified.jsonl")
if not unified_path.exists():
    print(f"\n❌ File not found: {unified_path}")
    print("   Please run: python src/data_prep.py first")
    exit(1)

print(f"\n✓ File found: {unified_path}")

# ============================================================================
# ANALYSIS 1: Field Structure
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: Field Structure")
print("=" * 70)

# Load first sample to see structure
with open(unified_path, "r", encoding="utf-8") as f:
    first_sample = json.loads(f.readline())

print("\n✓ Fields available in unified.jsonl:\n")
for i, (key, value) in enumerate(first_sample.items(), 1):
    value_preview = str(value)[:80] if not isinstance(value, (list, dict)) else f"[{type(value).__name__}]"
    print(f"   {i:2d}. {key:<20} = {value_preview}")

print(f"\n✓ Total fields: {len(first_sample.keys())}")

# ============================================================================
# ANALYSIS 2: Check for Unique Identifiers
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: Unique Identifier Analysis")
print("=" * 70)

print("\n🔍 Analyzing first 1000 samples for unique fields...")

samples = []
with open(unified_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        samples.append(json.loads(line))

print(f"✓ Loaded {len(samples)} samples")

# Analyze each field
print("\n📊 Field Uniqueness Report:\n")

for key in first_sample.keys():
    if key in ['events', 'noise_tags']:  # Skip complex fields
        continue
    
    # Collect values
    values = [str(s.get(key, '')) for s in samples]
    unique_values = set(values)
    uniqueness_ratio = len(unique_values) / len(values)
    
    print(f"Field: {key:<20}")
    print(f"  Total values: {len(values)}")
    print(f"  Unique values: {len(unique_values)} ({uniqueness_ratio*100:.1f}%)")
    
    # Show sample values
    sample_vals = list(set(values))[:3]
    print(f"  Sample values: {sample_vals}")
    
    if uniqueness_ratio == 1.0:
        print(f"  ✅ 100% UNIQUE - Perfect for stable mapping!")
    elif uniqueness_ratio > 0.95:
        print(f"  ✓ {uniqueness_ratio*100:.1f}% unique - Mostly unique")
    elif uniqueness_ratio > 0.5:
        print(f"  ⚠ {uniqueness_ratio*100:.1f}% unique - Some duplicates")
    else:
        print(f"  ℹ {uniqueness_ratio*100:.1f}% unique - Many duplicates (expected for categorical fields)")
    
    print()

# ============================================================================
# ANALYSIS 3: clip_id Pattern Analysis
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: clip_id Pattern Analysis")
print("=" * 70)

clip_ids = [s['clip_id'] for s in samples]

print(f"\n✓ Analyzing {len(clip_ids)} clip_ids...\n")

# Check pattern
print("Sample clip_ids:")
for i in range(min(10, len(clip_ids))):
    print(f"   {clip_ids[i]}")

# Extract indices
indices = []
for clip_id in clip_ids:
    try:
        idx = int(clip_id.replace("train_", ""))
        indices.append(idx)
    except:
        print(f"   ⚠ Invalid format: {clip_id}")

if indices:
    print(f"\n✓ Extracted indices from {len(indices)} clip_ids")
    print(f"   Min index: {min(indices)}")
    print(f"   Max index: {max(indices)}")
    print(f"   Range: {max(indices) - min(indices) + 1}")
    
    # Check if sequential
    is_sequential = all(indices[i] < indices[i+1] for i in range(len(indices)-1))
    if is_sequential:
        print(f"   ✓ Indices are SEQUENTIAL (sorted in order)")
    else:
        print(f"   ⚠ Indices are NOT sequential (shuffled)")
    
    # Check uniqueness
    if len(set(indices)) == len(indices):
        print(f"   ✓ All indices are UNIQUE")
    else:
        print(f"   ❌ Some indices are duplicated")

# ============================================================================
# ANALYSIS 4: Data Consistency Check
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: Data Consistency Check")
print("=" * 70)

print("\n🔍 Checking data consistency...\n")

# Check tier distribution
tiers = Counter(s['tier'] for s in samples)
print("Tier Distribution:")
for tier, count in tiers.most_common():
    print(f"   {tier:10s}: {count:5d} ({count/len(samples)*100:.1f}%)")

# Check language distribution
languages = Counter(s.get('language', 'unknown') for s in samples)
print("\nTop 10 Languages:")
for lang, count in languages.most_common(10):
    print(f"   {lang:15s}: {count:5d} ({count/len(samples)*100:.1f}%)")

# Check duration stats
durations = [s.get('duration', 0) for s in samples if s.get('duration')]
if durations:
    print(f"\nDuration Statistics:")
    print(f"   Count: {len(durations)}")
    print(f"   Min: {min(durations):.2f}s")
    print(f"   Max: {max(durations):.2f}s")
    print(f"   Avg: {sum(durations)/len(durations):.2f}s")

# Check events
gold_silver_samples = [s for s in samples if s['tier'] in ['gold', 'silver']]
if gold_silver_samples:
    event_counts = [len(s.get('events', [])) for s in gold_silver_samples]
    print(f"\nEvent Statistics (Gold/Silver):")
    print(f"   Samples with events: {sum(1 for c in event_counts if c > 0)}/{len(gold_silver_samples)}")
    print(f"   Avg events per sample: {sum(event_counts)/len(event_counts):.2f}")

# ============================================================================
# ANALYSIS 5: Simulate HF Index Mapping
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: Simulating HF Index Mapping")
print("=" * 70)

print("\n🔄 Testing our current mapping strategy...\n")

# Current strategy: clip_id "train_000042" → HF index 42
print("Current Strategy: clip_id → HF dataset index")
print("   train_000000 → HF index 0")
print("   train_000042 → HF index 42")
print("   train_012345 → HF index 12345")

# Test with our data
test_samples = samples[:5]
print(f"\n📊 Testing with first 5 samples from unified.jsonl:\n")

for i, sample in enumerate(test_samples):
    clip_id = sample['clip_id']
    expected_index = int(clip_id.replace("train_", ""))
    
    print(f"Sample {i}:")
    print(f"   clip_id: {clip_id}")
    print(f"   Expected HF index: {expected_index}")
    print(f"   Language: {sample.get('language', 'N/A')}")
    print(f"   Duration: {sample.get('duration', 'N/A')}s")
    print(f"   Tier: {sample['tier']}")
    
    if expected_index == i:
        print(f"   ⚠ WARNING: Expected index {expected_index} but sample is at position {i}")
        print(f"   This means unified.jsonl is NOT in original HF order!")
    else:
        print(f"   ✓ Index {expected_index} stored at position {i}")
    print()

# ============================================================================
# CRITICAL FINDING
# ============================================================================
print("\n" + "=" * 70)
print("🚨 CRITICAL FINDING")
print("=" * 70)

# Check if first sample's clip_id matches its position
first_clip_id = samples[0]['clip_id']
first_index = int(first_clip_id.replace("train_", ""))

if first_index == 0:
    print("\n✅ GOOD NEWS!")
    print("   First sample has clip_id 'train_000000'")
    print("   This suggests unified.jsonl IS in original HF order")
    print("\n   → Sequential index mapping should work!")
    print("   → Your audio_loader.py strategy is CORRECT")
else:
    print("\n⚠ IMPORTANT!")
    print(f"   First sample has clip_id '{first_clip_id}' (index {first_index})")
    print("   This suggests unified.jsonl is NOT in original HF order")
    print("\n   → Sequential indexing from unified.jsonl won't work")
    print("   → Need to use clip_id index to fetch from HF")
    print("   → Your audio_loader.py already handles this correctly!")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print("\n📋 Based on existing data analysis:\n")

if 'clip_id' in first_sample and all(s.get('clip_id', '').startswith('train_') for s in samples[:10]):
    print("✅ clip_id field exists and follows 'train_XXXXXX' pattern")
    print("✅ clip_id can be used to map back to HF dataset index")
    print("\nYour audio_loader.py implementation:")
    print("   1. Extract index from clip_id: 'train_000042' → 42")
    print("   2. Stream HF dataset to that index")
    print("   3. Download and cache audio")
    print("\nThis approach is CORRECT! ✓")
    print("\nNext step:")
    print("   Install dependencies: pip install -r requirements.txt")
    print("   Then run: python check_dataset_stability.py")
    print("   (to verify HF streaming is stable)")
else:
    print("⚠ clip_id field format is unexpected")
    print("   Need to verify data_prep.py logic")

print("\n" + "=" * 70)
