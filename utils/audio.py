# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Audio processing utilities for MSS-TF-Locoformer.
"""

from typing import Optional, Tuple

import torch
import torchaudio


def load_audio(
    path: str,
    sample_rate: int = 44100,
    mono: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Load audio file.
    
    Args:
        path (str): Path to audio file
        sample_rate (int): Target sample rate. Default: 44100
        mono (bool): Convert to mono. Default: False
        
    Returns:
        Tuple[torch.Tensor, int]: Audio tensor [C, T] and sample rate
    """
    audio, sr = torchaudio.load(path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    
    # Convert to mono
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    return audio, sample_rate


def save_audio(
    audio: torch.Tensor,
    path: str,
    sample_rate: int = 44100,
    normalize: bool = True,
) -> None:
    """Save audio to file.
    
    Args:
        audio (torch.Tensor): Audio tensor [C, T]
        path (str): Output file path
        sample_rate (int): Sample rate. Default: 44100
        normalize (bool): Normalize to [-1, 1]. Default: True
    """
    if normalize:
        # Normalize to [-1, 1]
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
    
    # Ensure audio is on CPU
    audio = audio.cpu()
    
    torchaudio.save(path, audio, sample_rate)


def normalize_audio(
    audio: torch.Tensor,
    target_level: float = -25.0,
) -> torch.Tensor:
    """Normalize audio to target RMS level in dB.
    
    Args:
        audio (torch.Tensor): Input audio
        target_level (float): Target level in dB. Default: -25.0
        
    Returns:
        torch.Tensor: Normalized audio
    """
    # Calculate RMS
    rms = torch.sqrt(torch.mean(audio ** 2))
    
    # Convert target level from dB to linear
    target_rms = 10 ** (target_level / 20)
    
    # Normalize
    if rms > 0:
        audio = audio * (target_rms / rms)
    
    # Clip to [-1, 1]
    audio = torch.clamp(audio, -1.0, 1.0)
    
    return audio


def compute_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 1024,
    window: str = 'hann',
) -> torch.Tensor:
    """Compute magnitude spectrogram.
    
    Args:
        audio (torch.Tensor): Audio tensor [B, T] or [T]
        n_fft (int): FFT size. Default: 2048
        hop_length (int): Hop length. Default: 1024
        window (str): Window type. Default: 'hann'
        
    Returns:
        torch.Tensor: Magnitude spectrogram [B, F, T] or [F, T]
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    window_tensor = torch.hann_window(n_fft, device=audio.device)
    
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_tensor,
        return_complex=True,
    )
    
    mag_spec = torch.abs(spec)
    
    if squeeze:
        mag_spec = mag_spec.squeeze(0)
    
    return mag_spec


def apply_gain(
    audio: torch.Tensor,
    gain_db: float,
) -> torch.Tensor:
    """Apply gain to audio in dB.
    
    Args:
        audio (torch.Tensor): Input audio
        gain_db (float): Gain in dB
        
    Returns:
        torch.Tensor: Audio with gain applied
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def mix_sources(
    sources: dict,
    weights: Optional[dict] = None,
) -> torch.Tensor:
    """Mix multiple audio sources.
    
    Args:
        sources (dict): Dictionary of audio tensors
        weights (dict, optional): Mixing weights for each source
        
    Returns:
        torch.Tensor: Mixed audio
    """
    if weights is None:
        weights = {k: 1.0 for k in sources.keys()}
    
    mixture = None
    for name, audio in sources.items():
        weight = weights.get(name, 1.0)
        weighted_audio = audio * weight
        
        if mixture is None:
            mixture = weighted_audio
        else:
            mixture = mixture + weighted_audio
    
    return mixture


def compute_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Compute audio separation metrics.
    
    Args:
        estimate (torch.Tensor): Estimated signal
        target (torch.Tensor): Target signal
        
    Returns:
        dict: Dictionary with metrics (SDR, SI-SDR)
    """
    eps = 1e-8
    
    # Flatten
    if estimate.ndim > 1:
        estimate = estimate.reshape(-1)
        target = target.reshape(-1)
    
    # Zero-mean
    estimate_zm = estimate - estimate.mean()
    target_zm = target - target.mean()
    
    # SI-SDR
    dot_product = torch.sum(estimate_zm * target_zm)
    target_energy = torch.sum(target_zm ** 2) + eps
    scale = dot_product / target_energy
    s_target = scale * target_zm
    
    signal_energy = torch.sum(s_target ** 2) + eps
    noise_energy = torch.sum((estimate_zm - s_target) ** 2) + eps
    si_sdr = 10 * torch.log10(signal_energy / noise_energy)
    
    # SDR
    target_energy = torch.sum(target ** 2) + eps
    noise_energy = torch.sum((estimate - target) ** 2) + eps
    sdr = 10 * torch.log10(target_energy / noise_energy)
    
    return {
        'si_sdr': si_sdr.item(),
        'sdr': sdr.item(),
    }


def pad_or_trim(
    audio: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Pad or trim audio to target length.
    
    Args:
        audio (torch.Tensor): Input audio [..., T]
        target_length (int): Target length
        
    Returns:
        torch.Tensor: Audio with target length
    """
    current_length = audio.shape[-1]
    
    if current_length < target_length:
        # Pad
        pad_length = target_length - current_length
        audio = torch.nn.functional.pad(audio, (0, pad_length))
    elif current_length > target_length:
        # Trim
        audio = audio[..., :target_length]
    
    return audio
