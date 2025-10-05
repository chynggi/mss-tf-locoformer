# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Dataset classes for Music Source Separation.

This module provides PyTorch Dataset classes for loading and processing
music datasets like MUSDB18 and MUSDB18-HQ.
"""

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, get_worker_info

try:
    import audiomentations as AU
except ImportError:  # pragma: no cover - optional dependency
    AU = None


@dataclass(frozen=True)
class TrackInfo:
    """Lightweight descriptor for a MUSDB track."""

    name: str
    directory: Path
    length: int
    sample_rate: int
    channels: int
    mixture_path: Path
    source_paths: Dict[str, Optional[Path]]


class MUSDBDataset(Dataset):
    """PyTorch Dataset for MUSDB18/MUSDB18-HQ with streaming-friendly loading.

    This implementation draws inspiration from
    https://github.com/ZFTurbo/Music-Source-Separation-Training and adds
    metadata caching, loudness-aware chunk selection, and optional advanced
    augmentations while preserving the simple API used across this project.
    """

    CACHE_VERSION = 2

    def __init__(
        self,
        root_dir: str,
        subset: str = 'train',
        sample_rate: int = 44100,
        segment_length: Optional[int] = None,
        sources: Optional[List[str]] = None,
        augmentation: bool = False,
        random_chunks: bool = True,
        min_loudness: float = 1e-4,
        max_chunk_attempts: int = 8,
        metadata_cache: bool = True,
        chunk_cache: Optional[str] = None,
        precompute_chunks: bool = False,
        chunk_hop: Optional[int] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        rebuild_mixture: bool = True,
        return_metadata: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.subset = subset
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        if self.segment_length is not None and self.segment_length <= 0:
            raise ValueError("segment_length must be positive")

        self.sources = list(sources or ['vocals', 'other'])
        self.augmentation = bool(augmentation) or isinstance(augmentation, dict)
        self.random_chunks = random_chunks
        self.min_loudness = max(0.0, float(min_loudness))
        self.max_chunk_attempts = max(1, int(max_chunk_attempts))
        self.precompute_chunks = precompute_chunks
        self.chunk_hop = chunk_hop
        self.rebuild_mixture = rebuild_mixture
        self.return_metadata = return_metadata
        self.verbose = verbose
        self.file_types = ('.wav', '.flac')
        self.segment_duration = (self.segment_length / self.sample_rate) if self.segment_length else None

        if isinstance(augmentation, dict):
            aug_cfg = dict(augmentation)
        else:
            aug_cfg = dict(augmentation_config or {})
        self._aug_config = self._default_aug_config()
        self._aug_config.update({k: v for k, v in aug_cfg.items() if v is not None})

        self._metadata_cache_base: Optional[Path] = None
        if metadata_cache:
            if isinstance(metadata_cache, (str, os.PathLike)):
                self._metadata_cache_base = Path(metadata_cache)
            else:
                self._metadata_cache_base = self.root_dir / '.cache'
            self._metadata_cache_base.mkdir(parents=True, exist_ok=True)

        if chunk_cache is None:
            self._chunk_cache_base = self._metadata_cache_base
        elif chunk_cache:
            self._chunk_cache_base = Path(chunk_cache)
            self._chunk_cache_base.mkdir(parents=True, exist_ok=True)
        else:
            self._chunk_cache_base = None

        subset_dir = self.root_dir / self.subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        self.subset_dir = subset_dir

        self.track_infos = self._load_track_metadata()
        if not self.track_infos:
            raise ValueError(f"No tracks found in {self.subset_dir}")

        self._chunk_index: Optional[List[Tuple[int, int]]] = None
        if self.precompute_chunks:
            if self.segment_length is None:
                raise ValueError("precompute_chunks requires segment_length to be set")
            self._chunk_index = self._load_or_build_chunk_index()

        self._log(f"Loaded {len(self.track_infos)} tracks from {self.subset} subset")
        if self._chunk_index is not None:
            self._log(f"Prepared {len(self._chunk_index)} chunk descriptors")

    def __len__(self) -> int:
        if self._chunk_index is not None:
            return len(self._chunk_index)
        return len(self.track_infos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.track_infos:
            raise IndexError("Dataset is empty")

        if self.segment_length is None:
            track = self.track_infos[idx % len(self.track_infos)]
            sample = self._load_full_track(track)
        else:
            if self._chunk_index is not None:
                track_idx, offset = self._chunk_index[idx % len(self._chunk_index)]
                track = self.track_infos[track_idx]
                sample = self._load_chunk(track, offset)
            elif self.random_chunks and self.subset == 'train':
                sample, track = self._sample_random_chunk()
            else:
                track = self.track_infos[idx % len(self.track_infos)]
                sample = self._load_chunk(track, 0)

        sample = self._ensure_mixture(sample, force_rebuild=False)
        mix_only_aug = False
        if self.augmentation and self.subset == 'train':
            sample, mix_only_aug = self._apply_augmentation(sample)
        if self.rebuild_mixture and not mix_only_aug:
            sample = self._ensure_mixture(sample, force_rebuild=True)
        else:
            sample = self._ensure_mixture(sample, force_rebuild=False)

        if self.return_metadata:
            sample['track_name'] = track.name
        return sample

    def _log(self, message: str) -> None:
        if not self.verbose:
            return
        worker = get_worker_info()
        if worker is not None and worker.id != 0:
            return
        print(message)

    def _metadata_cache_path(self) -> Optional[Path]:
        if self._metadata_cache_base is None:
            return None
        return self._metadata_cache_base / f"{self.subset}_metadata.pkl"

    def _chunk_cache_path(self) -> Optional[Path]:
        if self._chunk_cache_base is None:
            return None
        suffix = f"{self.subset}_sr{self.sample_rate}_seg{self.segment_length or 0}.pkl"
        return self._chunk_cache_base / suffix

    def _load_track_metadata(self) -> List[TrackInfo]:
        cache_path = self._metadata_cache_path()
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as handle:
                    payload = pickle.load(handle)
                if self._metadata_cache_matches(payload):
                    tracks = []
                    for item in payload['tracks']:
                        tracks.append(
                            TrackInfo(
                                name=item['name'],
                                directory=Path(item['directory']),
                                length=item['length'],
                                sample_rate=item['sample_rate'],
                                channels=item.get('channels', 2),
                                mixture_path=Path(item['mixture']),
                                source_paths={
                                    key: (Path(val) if val else None)
                                    for key, val in item['sources'].items()
                                },
                            )
                        )
                    return tracks
            except Exception:
                self._log(f"Metadata cache at {cache_path} is invalid, rebuilding...")

        tracks: List[TrackInfo] = []
        candidate_dirs = [d for d in sorted(self.subset_dir.iterdir()) if d.is_dir()]
        for directory in candidate_dirs:
            mixture_path = self._resolve_audio_path(directory, 'mixture')
            if mixture_path is None:
                self._log(f"Skipping {directory.name}: mixture not found")
                continue
            try:
                info = sf.info(str(mixture_path))
            except RuntimeError as exc:
                self._log(f"Skipping {directory.name}: {exc}")
                continue

            source_paths = {name: self._resolve_audio_path(directory, name) for name in self.sources}
            tracks.append(
                TrackInfo(
                    name=directory.name,
                    directory=directory,
                    length=info.frames,
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    mixture_path=mixture_path,
                    source_paths=source_paths,
                )
            )

        if cache_path:
            payload = {
                'version': self.CACHE_VERSION,
                'config': {
                    'root': str(self.root_dir),
                    'subset': self.subset,
                    'sample_rate': self.sample_rate,
                    'sources': tuple(self.sources),
                },
                'tracks': [
                    {
                        'name': track.name,
                        'directory': str(track.directory),
                        'length': track.length,
                        'sample_rate': track.sample_rate,
                        'channels': track.channels,
                        'mixture': str(track.mixture_path),
                        'sources': {k: (str(v) if v else None) for k, v in track.source_paths.items()},
                    }
                    for track in tracks
                ],
            }
            with open(cache_path, 'wb') as handle:
                pickle.dump(payload, handle)

        return tracks

    def _metadata_cache_matches(self, payload: Dict[str, Any]) -> bool:
        config = payload.get('config', {})
        return (
            payload.get('version') == self.CACHE_VERSION
            and config.get('root') == str(self.root_dir)
            and config.get('subset') == self.subset
            and config.get('sample_rate') == self.sample_rate
            and tuple(config.get('sources', ())) == tuple(self.sources)
            and 'tracks' in payload
        )

    def _load_or_build_chunk_index(self) -> List[Tuple[int, int]]:
        cache_path = self._chunk_cache_path()
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as handle:
                    payload = pickle.load(handle)
                if self._chunk_cache_matches(payload):
                    return payload['chunks']
            except Exception:
                self._log(f"Chunk cache at {cache_path} is invalid, rebuilding...")

        chunks = self._build_chunk_index()
        if cache_path:
            payload = {
                'version': self.CACHE_VERSION,
                'config': {
                    'sample_rate': self.sample_rate,
                    'segment_length': self.segment_length,
                    'chunk_hop': self.chunk_hop,
                    'min_loudness': self.min_loudness,
                    'sources': tuple(self.sources),
                },
                'chunks': chunks,
            }
            with open(cache_path, 'wb') as handle:
                pickle.dump(payload, handle)
        return chunks

    def _chunk_cache_matches(self, payload: Dict[str, Any]) -> bool:
        config = payload.get('config', {})
        return (
            payload.get('version') == self.CACHE_VERSION
            and config.get('sample_rate') == self.sample_rate
            and config.get('segment_length') == self.segment_length
            and config.get('chunk_hop') == self.chunk_hop
            and config.get('min_loudness') == self.min_loudness
            and tuple(config.get('sources', ())) == tuple(self.sources)
            and 'chunks' in payload
        )

    def _build_chunk_index(self) -> List[Tuple[int, int]]:
        chunks: List[Tuple[int, int]] = []
        for track_idx, track in enumerate(self.track_infos):
            chunk_frames = self._desired_frames_for_sr(track.sample_rate)
            if chunk_frames is None:
                chunks.append((track_idx, 0))
                continue

            hop_frames = self._chunk_hop_frames_for_sr(track.sample_rate)
            if track.length <= chunk_frames:
                offsets = [0]
            else:
                max_start = max(track.length - chunk_frames, 0)
                offsets = list(range(0, max_start + 1, hop_frames)) or [0]

            for offset in offsets:
                sample = self._load_chunk(track, offset)
                if self._passes_loudness(sample):
                    chunks.append((track_idx, offset))

        if not chunks:
            chunks = [(idx, 0) for idx in range(len(self.track_infos))]
        return chunks

    def _desired_frames_for_sr(self, sr: int) -> Optional[int]:
        if self.segment_length is None:
            return None
        frames = int(round(self.segment_length * sr / self.sample_rate))
        return max(frames, 1)

    def _chunk_hop_frames_for_sr(self, sr: int) -> int:
        if self.segment_length is None:
            return sr
        hop = self.chunk_hop if self.chunk_hop is not None else max(self.segment_length // 2, 1)
        frames = int(round(hop * sr / self.sample_rate))
        return max(frames, 1)

    def _random_offset(self, track: TrackInfo) -> int:
        chunk_frames = self._desired_frames_for_sr(track.sample_rate)
        if chunk_frames is None or track.length <= chunk_frames:
            return 0
        return random.randint(0, track.length - chunk_frames)

    def _sample_random_chunk(self) -> Tuple[Dict[str, torch.Tensor], TrackInfo]:
        for _ in range(self.max_chunk_attempts):
            track = random.choice(self.track_infos)
            offset = self._random_offset(track)
            sample = self._load_chunk(track, offset)
            if self._passes_loudness(sample):
                return sample, track
        track = random.choice(self.track_infos)
        return self._load_chunk(track, 0), track

    def _passes_loudness(self, sample: Dict[str, torch.Tensor]) -> bool:
        if self.min_loudness <= 0:
            return True
        stats: List[float] = []
        for stem in self.sources:
            audio = sample.get(stem)
            if audio is not None:
                stats.append(float(audio.abs().mean().item()))
        if not stats:
            mixture = sample.get('mixture')
            if mixture is not None:
                stats.append(float(mixture.abs().mean().item()))
        if not stats:
            return True
        return max(stats) >= self.min_loudness

    def _pad_or_trim(self, audio: torch.Tensor, target_frames: int) -> torch.Tensor:
        if audio.shape[-1] == target_frames:
            return audio
        if audio.shape[-1] > target_frames:
            return audio[..., :target_frames]
        pad_shape = list(audio.shape)
        pad_shape[-1] = target_frames - audio.shape[-1]
        padding = torch.zeros(*pad_shape, dtype=audio.dtype, device=audio.device)
        return torch.cat([audio, padding], dim=-1)

    def _load_full_track(self, track: TrackInfo) -> Dict[str, torch.Tensor]:
        sample: Dict[str, torch.Tensor] = {}
        mixture, sr = torchaudio.load(str(track.mixture_path))
        if sr != self.sample_rate:
            mixture = torchaudio.functional.resample(mixture, sr, self.sample_rate)
        sample['mixture'] = mixture.contiguous()

        for stem, path in track.source_paths.items():
            if path is None or not path.exists():
                sample[stem] = torch.zeros_like(sample['mixture'])
                continue
            audio, sr = torchaudio.load(str(path))
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            sample[stem] = self._pad_or_trim(audio.contiguous(), sample['mixture'].shape[-1])
        return sample

    def _load_chunk(self, track: TrackInfo, offset: int) -> Dict[str, torch.Tensor]:
        sample: Dict[str, torch.Tensor] = {}
        sample['mixture'] = self._read_segment(track.mixture_path, offset, track.sample_rate)
        for stem, path in track.source_paths.items():
            if path is None or not path.exists():
                sample[stem] = torch.zeros_like(sample['mixture'])
            else:
                sample[stem] = self._read_segment(path, offset, track.sample_rate)
        return sample

    def _read_segment(self, path: Path, offset_frames: Optional[int], reference_sr: int) -> torch.Tensor:
        with sf.SoundFile(str(path)) as handle:
            source_sr = handle.samplerate
            total_frames = handle.frames
            if self.segment_length is None:
                if offset_frames is not None:
                    offset = min(max(int(offset_frames), 0), total_frames)
                    handle.seek(offset)
                data = handle.read(dtype='float32', always_2d=True)
            else:
                desired_frames = self._desired_frames_for_sr(source_sr)
                if desired_frames is None:
                    desired_frames = total_frames
                offset = 0
                if offset_frames is not None:
                    offset = int(round(offset_frames * source_sr / reference_sr))
                    offset = min(max(offset, 0), max(total_frames - 1, 0))
                    handle.seek(offset)
                data = handle.read(frames=desired_frames, dtype='float32', always_2d=True)
                if desired_frames is not None and data.shape[0] < desired_frames:
                    pad = np.zeros((desired_frames - data.shape[0], data.shape[1]), dtype=np.float32)
                    data = np.concatenate([data, pad], axis=0)

        audio = torch.from_numpy(data.T)
        if self.segment_length is not None:
            audio = self._pad_or_trim(audio, self.segment_length)
        if source_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_sr, self.sample_rate)
            if self.segment_length is not None:
                audio = self._pad_or_trim(audio, self.segment_length)
        return audio.contiguous()

    def _resolve_audio_path(self, directory: Path, stem: str) -> Optional[Path]:
        for ext in self.file_types:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _ensure_mixture(self, sample: Dict[str, torch.Tensor], force_rebuild: bool) -> Dict[str, torch.Tensor]:
        mixture = sample.get('mixture')
        if mixture is None or force_rebuild:
            base: Optional[torch.Tensor] = None
            for stem in self.sources:
                audio = sample.get(stem)
                if audio is None:
                    continue
                if self.segment_length is not None:
                    audio = self._pad_or_trim(audio, self.segment_length)
                base = audio if base is None else base + audio
            if base is not None:
                if self.segment_length is not None:
                    base = self._pad_or_trim(base, self.segment_length)
                sample['mixture'] = base
        else:
            if self.segment_length is not None:
                sample['mixture'] = self._pad_or_trim(mixture, self.segment_length)
        return sample

    def _match_source_to_mixture(self, sample: Dict[str, torch.Tensor]) -> None:
        mix = sample.get('mixture')
        if mix is None:
            return
        target_frames = mix.shape[-1]
        for stem in self.sources:
            tensor = sample.get(stem)
            if tensor is None:
                continue
            sample[stem] = self._pad_or_trim(tensor.to(mix.device, mix.dtype), target_frames)

    def _apply_augmentation(self, sample: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
        mix_only_aug = False
        cfg = self._aug_config

        loud_range = cfg.get('loudness')
        if loud_range:
            low, high = loud_range
            for stem in self.sources:
                audio = sample.get(stem)
                if audio is None:
                    continue
                gain = random.uniform(low, high)
                sample[stem] = torch.clamp(audio * gain, -1.0, 1.0)

        channel_swap_prob = cfg.get('channel_swap_prob', 0.0)
        if channel_swap_prob > 0 and random.random() < channel_swap_prob:
            for key, audio in list(sample.items()):
                if isinstance(audio, torch.Tensor) and audio.dim() == 2 and audio.shape[0] == 2:
                    sample[key] = audio.flip(0)

        polarity_prob = cfg.get('polarity_prob', 0.0)
        if polarity_prob > 0 and random.random() < polarity_prob:
            for key, audio in list(sample.items()):
                if isinstance(audio, torch.Tensor):
                    sample[key] = -audio

        mixture = sample.get('mixture')
        if mixture is not None:
            expected_len = mixture.shape[-1]
            mix_device = mixture.device
            mix_dtype = mixture.dtype

            mp3_prob = cfg.get('mp3_prob', 0.0)
            if AU is not None and mp3_prob > 0 and random.random() < mp3_prob:
                bitrate_min, bitrate_max = cfg.get('mp3_bitrate', (96, 192))
                compressor = AU.Mp3Compression(
                    min_bitrate=int(bitrate_min),
                    max_bitrate=int(bitrate_max),
                    p=1.0,
                )
                mixture_np = mixture.detach().cpu().numpy().astype(np.float32)
                mixture_np = compressor(samples=mixture_np, sample_rate=self.sample_rate)
                mixture_np = mixture_np[..., :expected_len]
                sample['mixture'] = torch.from_numpy(mixture_np).to(device=mix_device, dtype=mix_dtype)
                mix_only_aug = True
            else:
                noise_prob = cfg.get('noise_prob', 0.0)
                if noise_prob > 0 and random.random() < noise_prob:
                    snr_min, snr_max = cfg.get('noise_snr', (25.0, 40.0))
                    snr = random.uniform(snr_min, snr_max)
                    noise = torch.randn_like(mixture)
                    signal_power = float(mixture.pow(2).mean().item())
                    if signal_power > 0:
                        noise_power = signal_power / (10 ** (snr / 10))
                        scale = torch.sqrt(torch.tensor(noise_power, dtype=mix_dtype, device=mix_device))
                        sample['mixture'] = torch.clamp(mixture + noise * scale, -1.0, 1.0)

            sample['mixture'] = self._pad_or_trim(sample['mixture'], expected_len)
            self._match_source_to_mixture(sample)

        for stem in self.sources:
            audio = sample.get(stem)
            if audio is not None:
                sample[stem] = torch.clamp(audio, -1.0, 1.0)

        return sample, mix_only_aug

    def _default_aug_config(self) -> Dict[str, Any]:
        return {
            'loudness': (0.7, 1.3),
            'channel_swap_prob': 0.5,
            'polarity_prob': 0.1,
            'mp3_prob': 0.2,
            'mp3_bitrate': (96, 192),
            'noise_prob': 0.1,
            'noise_snr': (25.0, 40.0),
        }


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
