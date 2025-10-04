# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Dataset classes for Music Source Separation.

This module provides PyTorch Dataset classes for loading and processing
music datasets like MUSDB18 and MUSDB18-HQ.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    """PyTorch Dataset for MUSDB18/MUSDB18-HQ.
    
    Args:
        root_dir (str): Root directory of MUSDB18 dataset
        subset (str): 'train' or 'test'
        sample_rate (int): Target sample rate. Default: 44100
        segment_length (Optional[int]): Length of audio segments in samples. 
            If None, use full tracks. Default: None
        sources (List[str]): List of source names to load. 
            Default: ['vocals', 'other'] (where 'other' = bass+drums+other)
        augmentation (bool): Apply data augmentation. Default: False
        random_chunks (bool): Extract random chunks from tracks. Default: True
        
    Note:
        This dataset expects a reduced file structure with only 3 files per track:
        - mixture.wav: full mix
        - vocals.wav: vocal track
        - other.wav: accompaniment (bass + drums + other instruments combined)
    """
    
    def __init__(
        self,
        root_dir: str,
        subset: str = 'train',
        sample_rate: int = 44100,
        segment_length: Optional[int] = None,
        sources: List[str] = ['vocals', 'other'],
        augmentation: bool = False,
        random_chunks: bool = True,
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.sources = sources
        self.augmentation = augmentation
        self.random_chunks = random_chunks
        
        # Get track directories
        subset_dir = self.root_dir / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
        self.track_dirs = sorted([d for d in subset_dir.iterdir() if d.is_dir()])
        
        if len(self.track_dirs) == 0:
            raise ValueError(f"No tracks found in {subset_dir}")
        
        print(f"Loaded {len(self.track_dirs)} tracks from {subset} subset")
    
    def __len__(self) -> int:
        return len(self.track_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a music track with its sources.
        
        Args:
            idx (int): Track index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'mixture': Mixed audio [2, T]
                - 'vocals': Vocals source [2, T]
                - 'other': Accompaniment (bass+drums+other) [2, T]
        """
        track_dir = self.track_dirs[idx]
        
        # Load mixture
        mixture_path = track_dir / "mixture.wav"
        mixture, sr = torchaudio.load(mixture_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            mixture = resampler(mixture)
        
        # Load sources
        sources_dict = {'mixture': mixture}
        for source_name in self.sources:
            source_path = track_dir / f"{source_name}.wav"
            if source_path.exists():
                source_audio, sr = torchaudio.load(source_path)
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    source_audio = resampler(source_audio)
                sources_dict[source_name] = source_audio
            else:
                # Create silent audio if source not found
                sources_dict[source_name] = torch.zeros_like(mixture)
        
        # Extract segment if needed
        if self.segment_length is not None:
            sources_dict = self._extract_segment(sources_dict)
        
        # Apply augmentation
        if self.augmentation and self.subset == 'train':
            sources_dict = self._apply_augmentation(sources_dict)
        
        # Convert to mono if needed (take first channel or average)
        # For MSS, we typically work with stereo, but can convert to mono
        # sources_dict = {k: v.mean(dim=0, keepdim=True) for k, v in sources_dict.items()}
        
        return sources_dict
    
    def _extract_segment(
        self,
        sources_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract a segment from the full track.
        
        Args:
            sources_dict (Dict[str, torch.Tensor]): Dictionary of audio tensors
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with extracted segments
        """
        mixture = sources_dict['mixture']
        _, total_length = mixture.shape
        
        if total_length < self.segment_length:
            # Pad if track is shorter than segment_length
            pad_length = self.segment_length - total_length
            for key in sources_dict:
                sources_dict[key] = torch.nn.functional.pad(
                    sources_dict[key],
                    (0, pad_length)
                )
        else:
            # Extract segment
            if self.random_chunks and self.subset == 'train':
                # Random start position
                start = random.randint(0, total_length - self.segment_length)
            else:
                # Take from beginning
                start = 0
            
            end = start + self.segment_length
            for key in sources_dict:
                sources_dict[key] = sources_dict[key][:, start:end]
        
        return sources_dict
    
    def _apply_augmentation(
        self,
        sources_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply data augmentation to sources.
        
        Args:
            sources_dict (Dict[str, torch.Tensor]): Dictionary of audio tensors
            
        Returns:
            Dict[str, torch.Tensor]: Augmented dictionary
        """
        # Random gain (volume) adjustment per source
        if random.random() > 0.5:
            for source_name in self.sources:
                if source_name in sources_dict:
                    gain = random.uniform(0.7, 1.3)
                    sources_dict[source_name] = sources_dict[source_name] * gain
            
            # Recreate mixture from augmented sources
            sources_dict['mixture'] = sum([
                sources_dict[s] for s in self.sources if s in sources_dict
            ])
        
        # Channel swapping
        if random.random() > 0.5:
            for key in sources_dict:
                if sources_dict[key].shape[0] == 2:
                    sources_dict[key] = sources_dict[key].flip(0)
        
        # Polarity inversion
        if random.random() > 0.9:
            for key in sources_dict:
                sources_dict[key] = -sources_dict[key]
        
        return sources_dict


class SimpleAudioDataset(Dataset):
    """Simple audio dataset for custom audio files.
    
    Args:
        audio_dir (str): Directory containing audio files
        sample_rate (int): Target sample rate. Default: 44100
        segment_length (Optional[int]): Length of segments. Default: None
        extensions (List[str]): Valid audio file extensions. 
            Default: ['.wav', '.mp3', '.flac']
    """
    
    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 44100,
        segment_length: Optional[int] = None,
        extensions: List[str] = ['.wav', '.mp3', '.flac'],
    ):
        super().__init__()
        
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        
        # Find all audio files
        self.audio_files = []
        for ext in extensions:
            self.audio_files.extend(self.audio_dir.glob(f"**/*{ext}"))
        
        self.audio_files = sorted(self.audio_files)
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load an audio file.
        
        Args:
            idx (int): File index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'audio' and 'path'
        """
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Extract segment if needed
        if self.segment_length is not None and audio.shape[-1] > self.segment_length:
            start = 0
            audio = audio[:, start:start + self.segment_length]
        
        return {
            'audio': audio,
            'path': str(audio_path),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching audio data.
    
    Args:
        batch (List[Dict[str, torch.Tensor]]): List of samples
        
    Returns:
        Dict[str, torch.Tensor]: Batched tensors
    """
    # Find max length in batch
    max_length = max([item['mixture'].shape[-1] for item in batch])
    
    # Pad all items to max length
    batched = {}
    keys = batch[0].keys()
    
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            padded_items = []
            for item in batch:
                tensor = item[key]
                if tensor.shape[-1] < max_length:
                    pad_length = max_length - tensor.shape[-1]
                    tensor = torch.nn.functional.pad(tensor, (0, pad_length))
                padded_items.append(tensor)
            batched[key] = torch.stack(padded_items, dim=0)
        else:
            batched[key] = [item[key] for item in batch]
    
    return batched


def normalize_audio(audio: torch.Tensor, target_level: float = -25.0) -> torch.Tensor:
    """Normalize audio to target level in dB.
    
    Args:
        audio (torch.Tensor): Input audio tensor
        target_level (float): Target level in dB. Default: -25.0
        
    Returns:
        torch.Tensor: Normalized audio
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(audio ** 2))
    
    # Convert target level from dB to linear
    target_rms = 10 ** (target_level / 20)
    
    # Normalize
    if rms > 0:
        audio = audio * (target_rms / rms)
    
    # Clip to [-1, 1]
    audio = torch.clamp(audio, -1.0, 1.0)
    
    return audio
