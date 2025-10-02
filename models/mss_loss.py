# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Loss functions for Music Source Separation.

This module provides various loss functions commonly used in MSS,
including SI-SDR, L1, L2, and combination losses.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSSLoss(nn.Module):
    """Combined loss function for Music Source Separation.
    
    This loss combines multiple objectives:
    - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    - L1 loss (time domain)
    - Spectral L1 loss (frequency domain)
    
    Args:
        loss_type (str): Type of loss ('si_sdr', 'l1', 'l2', 'combined'). Default: 'combined'
        si_sdr_weight (float): Weight for SI-SDR loss. Default: 1.0
        l1_weight (float): Weight for L1 loss. Default: 0.1
        spectral_weight (float): Weight for spectral loss. Default: 0.1
        eps (float): Small constant for numerical stability. Default: 1e-8
    """
    
    def __init__(
        self,
        loss_type: str = 'combined',
        si_sdr_weight: float = 1.0,
        l1_weight: float = 0.1,
        spectral_weight: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.si_sdr_weight = si_sdr_weight
        self.l1_weight = l1_weight
        self.spectral_weight = spectral_weight
        self.eps = eps
        
        # Loss components
        self.si_sdr_loss = SISDRLoss(eps=eps)
        self.l1_loss = nn.L1Loss()
        self.spectral_loss = SpectralLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute loss between predictions and targets.
        
        Args:
            predictions (Dict[str, torch.Tensor]): Predicted sources
                {'vocals': [B, T], 'drums': [B, T], 'bass': [B, T], 'other': [B, T]}
            targets (Dict[str, torch.Tensor]): Ground truth sources
                Same format as predictions
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary with total loss and individual components
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Compute loss for each source
        for source_name in predictions.keys():
            if source_name not in targets:
                continue
            
            pred = predictions[source_name]
            target = targets[source_name]
            
            # SI-SDR loss
            if self.loss_type in ['si_sdr', 'combined']:
                si_sdr = self.si_sdr_loss(pred, target)
                loss_dict[f'{source_name}_si_sdr'] = si_sdr
                total_loss += self.si_sdr_weight * si_sdr
            
            # L1 loss
            if self.loss_type in ['l1', 'combined']:
                l1 = self.l1_loss(pred, target)
                loss_dict[f'{source_name}_l1'] = l1
                total_loss += self.l1_weight * l1
            
            # L2 loss
            if self.loss_type == 'l2':
                l2 = F.mse_loss(pred, target)
                loss_dict[f'{source_name}_l2'] = l2
                total_loss += l2
            
            # Spectral loss
            if self.loss_type == 'combined' and self.spectral_weight > 0:
                spec_loss = self.spectral_loss(pred, target)
                loss_dict[f'{source_name}_spectral'] = spec_loss
                total_loss += self.spectral_weight * spec_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict


class SISDRLoss(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.
    
    SI-SDR is a popular metric for speech and music separation that is
    invariant to the scaling of the signal.
    
    Args:
        eps (float): Small constant for numerical stability. Default: 1e-8
        return_individual (bool): Return individual SI-SDR values. Default: False
    """
    
    def __init__(self, eps: float = 1e-8, return_individual: bool = False):
        super().__init__()
        self.eps = eps
        self.return_individual = return_individual
    
    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SI-SDR loss.
        
        Args:
            estimate (torch.Tensor): Estimated signal [B, T] or [B, C, T]
            target (torch.Tensor): Target signal [B, T] or [B, C, T]
            
        Returns:
            torch.Tensor: Negative SI-SDR (lower is better for loss)
        """
        # Ensure same shape
        assert estimate.shape == target.shape, f"Shape mismatch: {estimate.shape} vs {target.shape}"
        
        # Flatten to [B, -1]
        if estimate.ndim > 2:
            estimate = estimate.reshape(estimate.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
        
        # Zero-mean normalization
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # Compute scaling factor
        # s_target = <estimate, target> * target / ||target||^2
        dot_product = torch.sum(estimate * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        scale = dot_product / target_energy
        
        # Scaled target
        s_target = scale * target
        
        # Compute SI-SDR
        # SI-SDR = 10 * log10(||s_target||^2 / ||estimate - s_target||^2)
        signal_energy = torch.sum(s_target ** 2, dim=-1) + self.eps
        noise_energy = torch.sum((estimate - s_target) ** 2, dim=-1) + self.eps
        si_sdr = 10 * torch.log10(signal_energy / noise_energy)
        
        if self.return_individual:
            return -si_sdr  # Negative because we want to minimize
        else:
            return -si_sdr.mean()  # Average across batch


class SpectralLoss(nn.Module):
    """Spectral loss in STFT domain.
    
    This loss computes the L1 distance between magnitude spectrograms.
    
    Args:
        n_fft (int): FFT size. Default: 2048
        hop_length (int): Hop length. Default: 1024
        log_scale (bool): Use log-scale magnitudes. Default: True
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 1024,
        log_scale: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_scale = log_scale
    
    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spectral loss.
        
        Args:
            estimate (torch.Tensor): Estimated signal [B, T] or [B, C, T]
            target (torch.Tensor): Target signal [B, T] or [B, C, T]
            
        Returns:
            torch.Tensor: Spectral loss value
        """
        # Compute STFT
        window = torch.hann_window(self.n_fft, device=estimate.device)
        
        # Handle multichannel
        if estimate.ndim == 3:
            estimate = estimate.reshape(-1, estimate.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        
        estimate_stft = torch.stft(
            estimate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        
        target_stft = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        
        # Compute magnitudes
        estimate_mag = torch.abs(estimate_stft)
        target_mag = torch.abs(target_stft)
        
        # Log scale
        if self.log_scale:
            estimate_mag = torch.log1p(estimate_mag)
            target_mag = torch.log1p(target_mag)
        
        # L1 loss
        loss = F.l1_loss(estimate_mag, target_mag)
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss.
    
    Computes spectral loss at multiple resolutions for better
    frequency coverage.
    
    Args:
        fft_sizes (list): List of FFT sizes. Default: [512, 1024, 2048]
        hop_sizes (list): List of hop sizes. Default: [256, 512, 1024]
        win_sizes (list): List of window sizes. Default: [512, 1024, 2048]
    """
    
    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [256, 512, 1024],
        win_sizes: list = [512, 1024, 2048],
    ):
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        
        self.losses = nn.ModuleList([
            SpectralLoss(n_fft=fft_size, hop_length=hop_size)
            for fft_size, hop_size in zip(fft_sizes, hop_sizes)
        ])
    
    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-resolution spectral loss.
        
        Args:
            estimate (torch.Tensor): Estimated signal
            target (torch.Tensor): Target signal
            
        Returns:
            torch.Tensor: Average spectral loss across resolutions
        """
        total_loss = 0.0
        for loss_fn in self.losses:
            total_loss += loss_fn(estimate, target)
        
        return total_loss / len(self.losses)


def compute_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Signal-to-Distortion Ratio (SDR).
    
    Args:
        estimate (torch.Tensor): Estimated signal [B, T]
        target (torch.Tensor): Target signal [B, T]
        eps (float): Small constant for stability
        
    Returns:
        torch.Tensor: SDR values [B]
    """
    # Flatten
    if estimate.ndim > 2:
        estimate = estimate.reshape(estimate.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    
    # Compute SDR
    signal_energy = torch.sum(target ** 2, dim=-1) + eps
    noise_energy = torch.sum((estimate - target) ** 2, dim=-1) + eps
    sdr = 10 * torch.log10(signal_energy / noise_energy)
    
    return sdr


def compute_si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Scale-Invariant SDR (SI-SDR).
    
    Args:
        estimate (torch.Tensor): Estimated signal [B, T]
        target (torch.Tensor): Target signal [B, T]
        eps (float): Small constant for stability
        
    Returns:
        torch.Tensor: SI-SDR values [B]
    """
    # Flatten
    if estimate.ndim > 2:
        estimate = estimate.reshape(estimate.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # Compute scaling factor
    dot_product = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    scale = dot_product / target_energy
    
    # Scaled target
    s_target = scale * target
    
    # Compute SI-SDR
    signal_energy = torch.sum(s_target ** 2, dim=-1) + eps
    noise_energy = torch.sum((estimate - s_target) ** 2, dim=-1) + eps
    si_sdr = 10 * torch.log10(signal_energy / noise_energy)
    
    return si_sdr
