"""
audio_loader.py
Lazy audio loading utility with local caching for the Vaani-Noise-Event-Dataset.

Solves the disk space problem by:
1. Streaming only the required audio sample from HuggingFace on-demand
2. Caching downloaded files locally for reuse
3. Mapping clip_id back to the original HF dataset index

Usage:
    from src.audio_loader import AudioLoader
    
    loader = AudioLoader(cache_dir="data/cache/audio_clips")
    audio_array, sr = loader.load_audio("train_000042")
    
    # Or with batch loading:
    for clip_id in ["train_000001", "train_000002"]:
        audio, sr = loader.load_audio(clip_id)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import torchaudio
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from tqdm import tqdm


class AudioLoader:
    """
    On-demand audio loader with local caching for Vaani dataset.
    
    Features:
    - Lazy loading: only downloads audio when requested
    - Local caching: saves .wav files to avoid re-downloading
    - Stable mapping: uses clip_id (train_XXXXXX) to fetch from HF dataset
    - Memory efficient: streams from HF instead of loading entire dataset
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache/audio_clips",
        dataset_name: str = "ARTPARK-IISc/Vaani-Noise-Event-Dataset",
        use_streaming: bool = True,
        target_sr: Optional[int] = 16000
    ):
        """
        Initialize the AudioLoader.
        
        Args:
            cache_dir: Directory to store cached audio files
            dataset_name: HuggingFace dataset identifier
            use_streaming: If True, stream dataset; if False, download entire dataset
            target_sr: Target sampling rate (None to keep original)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = dataset_name
        self.use_streaming = use_streaming
        self.target_sr = target_sr
        
        # Load HF token from environment
        load_dotenv()
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError(
                "HF_TOKEN not found in .env file. "
                "Copy .env.example to .env and add your token."
            )
        
        # Lazy-load the dataset (streaming mode to avoid downloading everything)
        self._dataset = None
        self._dataset_iter = None
        self._index_cache = {}  # Map clip_id -> HF index for faster lookup
        
        print(f"AudioLoader initialized. Cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, clip_id: str) -> Path:
        """Generate cache file path for a given clip_id."""
        # Use clip_id as filename (e.g., train_000042.wav)
        return self.cache_dir / f"{clip_id}.wav"
    
    def _load_from_cache(self, clip_id: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio from local cache if it exists.
        
        Returns:
            (audio_array, sample_rate) if cached, None otherwise
        """
        cache_path = self._get_cache_path(clip_id)
        
        if cache_path.exists():
            try:
                audio, sr = sf.read(cache_path)
                return audio, sr
            except Exception as e:
                print(f"Warning: Failed to load cached file {cache_path}: {e}")
                # Delete corrupted cache file
                cache_path.unlink(missing_ok=True)
                return None
        
        return None
    
    def _save_to_cache(self, clip_id: str, audio: np.ndarray, sr: int):
        """Save audio array to local cache."""
        cache_path = self._get_cache_path(clip_id)
        try:
            sf.write(cache_path, audio, sr)
        except Exception as e:
            print(f"Warning: Failed to save to cache {cache_path}: {e}")
    
    def _get_dataset(self):
        """Lazy-load the HuggingFace dataset."""
        if self._dataset is None:
            print("Loading Vaani dataset from HuggingFace (streaming mode)...")
            self._dataset = load_dataset(
                self.dataset_name,
                token=self.token,
                streaming=self.use_streaming
            )
        return self._dataset
    
    def _extract_index_from_clip_id(self, clip_id: str) -> int:
        """
        Extract the numeric index from clip_id.
        
        Examples:
            "train_000042" -> 42
            "train_012345" -> 12345
        """
        try:
            # Remove "train_" prefix and convert to int
            return int(clip_id.replace("train_", ""))
        except ValueError:
            raise ValueError(f"Invalid clip_id format: {clip_id}. Expected 'train_XXXXXX'")
    
    def _fetch_from_hf(self, clip_id: str) -> Tuple[np.ndarray, int]:
        """
        Fetch audio from HuggingFace dataset by clip_id.
        
        Returns:
            (audio_array, sample_rate)
        """
        dataset = self._get_dataset()
        train_split = dataset["train"]
        
        # Extract index from clip_id
        target_index = self._extract_index_from_clip_id(clip_id)
        
        # For streaming datasets, we need to iterate to the target index
        if self.use_streaming:
            # Cast audio column to decode audio data
            train_split = train_split.cast_column("audio", Audio(decode=True))
            
            for i, sample in enumerate(train_split):
                if i == target_index:
                    audio_data = sample["audio"]
                    audio_array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data["sampling_rate"]
                    
                    # Resample if needed
                    if self.target_sr is not None and sr != self.target_sr:
                        audio_array, sr = self._resample(audio_array, sr, self.target_sr)
                    
                    return audio_array, sr
                
                # Early exit if we've passed the target (dataset is sequential)
                if i > target_index:
                    break
            
            raise ValueError(f"Could not find clip_id {clip_id} (index {target_index}) in dataset")
        
        else:
            # Non-streaming: direct index access
            sample = train_split[target_index]
            audio_data = sample["audio"]
            audio_array = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data["sampling_rate"]
            
            if self.target_sr is not None and sr != self.target_sr:
                audio_array, sr = self._resample(audio_array, sr, self.target_sr)
            
            return audio_array, sr
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
        """Resample audio to target sampling rate using torchaudio."""
        import torch
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Add channel dimension if mono
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Resample
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio_resampled = resampler(audio_tensor)
        
        # Convert back to numpy and squeeze
        audio_np = audio_resampled.squeeze().numpy()
        
        return audio_np, target_sr
    
    def load_audio(self, clip_id: str, force_reload: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load audio for a given clip_id with caching.
        
        Args:
            clip_id: Clip identifier (e.g., "train_000042")
            force_reload: If True, bypass cache and re-download from HF
            
        Returns:
            (audio_array, sample_rate) tuple
            
        Example:
            >>> loader = AudioLoader()
            >>> audio, sr = loader.load_audio("train_000042")
            >>> print(audio.shape, sr)
            (72800,) 16000
        """
        # Check cache first (unless force_reload is True)
        if not force_reload:
            cached = self._load_from_cache(clip_id)
            if cached is not None:
                return cached
        
        # Fetch from HuggingFace
        print(f"Fetching {clip_id} from HuggingFace...")
        audio, sr = self._fetch_from_hf(clip_id)
        
        # Save to cache
        self._save_to_cache(clip_id, audio, sr)
        
        return audio, sr
    
    def batch_load(self, clip_ids: list, show_progress: bool = True) -> dict:
        """
        Load multiple audio clips with progress bar.
        
        Args:
            clip_ids: List of clip identifiers
            show_progress: Show tqdm progress bar
            
        Returns:
            Dictionary mapping clip_id -> (audio_array, sample_rate)
        """
        results = {}
        iterator = tqdm(clip_ids, desc="Loading audio") if show_progress else clip_ids
        
        for clip_id in iterator:
            try:
                results[clip_id] = self.load_audio(clip_id)
            except Exception as e:
                print(f"Error loading {clip_id}: {e}")
                results[clip_id] = None
        
        return results
    
    def clear_cache(self, clip_id: Optional[str] = None):
        """
        Clear cached audio files.
        
        Args:
            clip_id: If provided, clear only this clip; otherwise clear all cache
        """
        if clip_id is not None:
            cache_path = self._get_cache_path(clip_id)
            cache_path.unlink(missing_ok=True)
            print(f"Cleared cache for {clip_id}")
        else:
            for cache_file in self.cache_dir.glob("*.wav"):
                cache_file.unlink()
            print(f"Cleared all cached audio files from {self.cache_dir}")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the cache directory."""
        cache_files = list(self.cache_dir.glob("*.wav"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        
        return {
            "num_cached": len(cache_files),
            "total_size_mb": round(total_size_mb, 2),
            "cache_dir": str(self.cache_dir)
        }


# Convenience function for quick usage
def load_audio_for_clip(clip_id: str, cache_dir: str = "data/cache/audio_clips") -> Tuple[np.ndarray, int]:
    """
    Quick utility function to load a single audio clip.
    
    Args:
        clip_id: Clip identifier (e.g., "train_000042")
        cache_dir: Cache directory path
        
    Returns:
        (audio_array, sample_rate) tuple
    """
    loader = AudioLoader(cache_dir=cache_dir)
    return loader.load_audio(clip_id)


if __name__ == "__main__":
    # Example usage and testing
    print("=== AudioLoader Test ===\n")
    
    # Initialize loader
    loader = AudioLoader(cache_dir="data/cache/audio_clips", target_sr=16000)
    
    # Test loading a few clips
    test_clip_ids = ["train_000001", "train_000002", "train_000005"]
    
    print("Loading test clips...")
    for clip_id in test_clip_ids:
        try:
            audio, sr = loader.load_audio(clip_id)
            print(f"{clip_id}: shape={audio.shape}, sr={sr}, duration={len(audio)/sr:.2f}s")
        except Exception as e:
            print(f"Error loading {clip_id}: {e}")
    
    # Show cache stats
    print("\nCache statistics:")
    stats = loader.get_cache_stats()
    print(f"  Cached files: {stats['num_cached']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    print(f"  Cache directory: {stats['cache_dir']}")
