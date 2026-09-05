"""
dataset.py
PyTorch Dataset wrapper for Vaani Noise Event Detection.

Integrates with audio_loader.py for lazy loading and local caching.

Usage:
    from src.dataset import VaaniNoiseDataset
    from torch.utils.data import DataLoader
    
    # Load dataset
    train_dataset = VaaniNoiseDataset(
        jsonl_path="data/processed/train_split.jsonl",
        cache_dir="data/cache/audio_clips"
    )
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Iterate
    for batch in train_loader:
        audio, events, metadata = batch
        # Your training code here
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

from .audio_loader import AudioLoader


class VaaniNoiseDataset(Dataset):
    """
    PyTorch Dataset for Vaani Noise Event Detection.
    
    Features:
    - Lazy audio loading with caching (via AudioLoader)
    - Supports Gold/Silver/Bronze annotation tiers
    - Returns audio waveforms + event annotations + metadata
    - Memory-efficient: audio loaded on-demand during training
    """
    
    def __init__(
        self,
        jsonl_path: str,
        cache_dir: str = "data/cache/audio_clips",
        target_sr: int = 16000,
        max_duration: Optional[float] = None,
        tier_filter: Optional[List[str]] = None,
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            jsonl_path: Path to JSONL file (train_split.jsonl or val_split.jsonl)
            cache_dir: Directory for caching audio files
            target_sr: Target sampling rate
            max_duration: Maximum audio duration in seconds (for padding/truncation)
            tier_filter: List of tiers to include (e.g., ["gold", "silver"])
            transform: Optional audio transformation/augmentation function
        """
        self.jsonl_path = Path(jsonl_path)
        self.cache_dir = cache_dir
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.tier_filter = tier_filter
        self.transform = transform
        
        # Initialize audio loader
        self.audio_loader = AudioLoader(
            cache_dir=cache_dir,
            target_sr=target_sr
        )
        
        # Load metadata from JSONL
        self.samples = self._load_metadata()
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
        if tier_filter:
            print(f"Filtered to tiers: {tier_filter}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load and optionally filter samples from JSONL."""
        samples = []
        
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                
                # Apply tier filter if specified
                if self.tier_filter is not None:
                    if record["tier"] not in self.tier_filter:
                        continue
                
                samples.append(record)
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - 'audio': torch.Tensor of shape (num_samples,)
            - 'sample_rate': int
            - 'events': List of event dictionaries with onset/offset/category
            - 'clip_id': str
            - 'tier': str (gold/silver/bronze)
            - 'language': str
            - 'duration': float
            - 'transcript': str
            - 'noise_tags': List (for bronze tier)
        """
        sample = self.samples[idx]
        clip_id = sample["clip_id"]
        
        # Load audio using lazy loader with caching
        try:
            audio, sr = self.audio_loader.load_audio(clip_id)
        except Exception as e:
            print(f"Error loading audio for {clip_id}: {e}")
            # Return empty audio as fallback
            audio = np.zeros(int(self.target_sr * (self.max_duration or 10)), dtype=np.float32)
            sr = self.target_sr
        
        # Pad or truncate to max_duration if specified
        if self.max_duration is not None:
            target_length = int(self.max_duration * sr)
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)))
            elif len(audio) > target_length:
                # Truncate
                audio = audio[:target_length]
        
        # Apply transforms if provided (e.g., augmentation)
        if self.transform is not None:
            audio = self.transform(audio)
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        return {
            'audio': audio_tensor,
            'sample_rate': sr,
            'events': sample.get('events', []),
            'clip_id': clip_id,
            'tier': sample['tier'],
            'language': sample.get('language', 'unknown'),
            'duration': sample.get('duration', len(audio) / sr),
            'transcript': sample.get('transcript', ''),
            'noise_tags': sample.get('noise_tags', [])
        }
    
    def get_tier_distribution(self) -> Dict[str, int]:
        """Get the distribution of annotation tiers in the dataset."""
        from collections import Counter
        return dict(Counter(s['tier'] for s in self.samples))
    
    def get_language_distribution(self) -> Dict[str, int]:
        """Get the distribution of languages in the dataset."""
        from collections import Counter
        return dict(Counter(s.get('language', 'unknown') for s in self.samples))
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the audio cache."""
        return self.audio_loader.get_cache_stats()


def collate_fn_vaani(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.
    
    Handles variable-length audio and events.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with:
        - 'audio': torch.Tensor of shape (batch_size, max_length)
        - 'sample_rate': int (assumes all have same SR)
        - 'events': List of event lists (not tensorized, varies per sample)
        - 'clip_ids': List of clip IDs
        - 'tiers': List of tiers
        - 'languages': List of languages
        - 'durations': torch.Tensor of shape (batch_size,)
    """
    # Find max audio length in batch
    max_length = max(sample['audio'].shape[0] for sample in batch)
    batch_size = len(batch)
    
    # Pad all audio to same length
    audio_padded = torch.zeros(batch_size, max_length)
    for i, sample in enumerate(batch):
        audio_len = sample['audio'].shape[0]
        audio_padded[i, :audio_len] = sample['audio']
    
    return {
        'audio': audio_padded,
        'sample_rate': batch[0]['sample_rate'],  # Assume all same SR
        'events': [sample['events'] for sample in batch],
        'clip_ids': [sample['clip_id'] for sample in batch],
        'tiers': [sample['tier'] for sample in batch],
        'languages': [sample['language'] for sample in batch],
        'durations': torch.tensor([sample['duration'] for sample in batch]),
        'transcripts': [sample['transcript'] for sample in batch],
        'noise_tags': [sample['noise_tags'] for sample in batch]
    }


if __name__ == "__main__":
    # Example usage and testing
    print("=== VaaniNoiseDataset Test ===\n")
    
    # Test with training split
    train_dataset = VaaniNoiseDataset(
        jsonl_path="data/processed/train_split.jsonl",
        cache_dir="data/cache/audio_clips",
        target_sr=16000,
        max_duration=10.0  # Pad/truncate to 10 seconds
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Tier distribution: {train_dataset.get_tier_distribution()}")
    print(f"Language distribution (top 10): {dict(list(train_dataset.get_language_distribution().items())[:10])}")
    
    # Test loading a single sample
    print("\n--- Sample 0 ---")
    sample = train_dataset[0]
    print(f"Clip ID: {sample['clip_id']}")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Sample rate: {sample['sample_rate']}")
    print(f"Tier: {sample['tier']}")
    print(f"Language: {sample['language']}")
    print(f"Duration: {sample['duration']:.2f}s")
    print(f"Events: {sample['events']}")
    print(f"Transcript: {sample['transcript'][:100]}...")
    
    # Test with DataLoader
    print("\n--- Testing DataLoader ---")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_vaani,
        num_workers=0  # Set >0 for multiprocessing
    )
    
    batch = next(iter(loader))
    print(f"Batch audio shape: {batch['audio'].shape}")
    print(f"Batch clip_ids: {batch['clip_ids']}")
    print(f"Batch tiers: {batch['tiers']}")
    
    # Cache stats
    print("\n--- Cache Stats ---")
    stats = train_dataset.get_cache_stats()
    print(f"Cached files: {stats['num_cached']}")
    print(f"Total cache size: {stats['total_size_mb']} MB")
