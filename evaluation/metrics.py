# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation metrics for Music Source Separation.
"""

from typing import Dict, Optional

import torch
import numpy as np


def compute_si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        estimate (torch.Tensor): Estimated signal
        target (torch.Tensor): Target signal
        eps (float): Small constant for stability
        
    Returns:
        float: SI-SDR value in dB
    """
    # Convert to numpy if needed
    if isinstance(estimate, torch.Tensor):
        estimate = estimate.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten
    estimate = estimate.flatten()
    target = target.flatten()
    
    # Zero-mean
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    
    # Compute scaling factor
    dot_product = np.dot(estimate, target)
    target_energy = np.dot(target, target) + eps
    scale = dot_product / target_energy
    
    # Scaled target
    s_target = scale * target
    
    # Compute SI-SDR
    signal_energy = np.dot(s_target, s_target) + eps
    noise_energy = np.dot(estimate - s_target, estimate - s_target) + eps
    si_sdr = 10 * np.log10(signal_energy / noise_energy)
    
    return float(si_sdr)


def compute_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute Signal-to-Distortion Ratio (SDR).
    
    Args:
        estimate (torch.Tensor): Estimated signal
        target (torch.Tensor): Target signal
        eps (float): Small constant for stability
        
    Returns:
        float: SDR value in dB
    """
    if isinstance(estimate, torch.Tensor):
        estimate = estimate.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    estimate = estimate.flatten()
    target = target.flatten()
    
    signal_energy = np.dot(target, target) + eps
    noise_energy = np.dot(estimate - target, estimate - target) + eps
    sdr = 10 * np.log10(signal_energy / noise_energy)
    
    return float(sdr)


def compute_sar(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute Signal-to-Artifacts Ratio (SAR).
    
    Args:
        estimate (torch.Tensor): Estimated signal
        target (torch.Tensor): Target signal
        eps (float): Small constant for stability
        
    Returns:
        float: SAR value in dB
    """
    if isinstance(estimate, torch.Tensor):
        estimate = estimate.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    estimate = estimate.flatten()
    target = target.flatten()
    
    # Compute projection
    dot_product = np.dot(estimate, target)
    target_energy = np.dot(target, target) + eps
    scale = dot_product / target_energy
    s_target = scale * target
    
    # Artifacts
    e_artifact = estimate - s_target
    
    # SAR
    signal_energy = np.dot(s_target, s_target) + eps
    artifact_energy = np.dot(e_artifact, e_artifact) + eps
    sar = 10 * np.log10(signal_energy / artifact_energy)
    
    return float(sar)


def compute_sir(
    estimate: torch.Tensor,
    target: torch.Tensor,
    references: Optional[list] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Signal-to-Interference Ratio (SIR).
    
    Args:
        estimate (torch.Tensor): Estimated signal
        target (torch.Tensor): Target signal
        references (list, optional): Other source references for interference
        eps (float): Small constant for stability
        
    Returns:
        float: SIR value in dB
    """
    if isinstance(estimate, torch.Tensor):
        estimate = estimate.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    estimate = estimate.flatten()
    target = target.flatten()
    
    # Compute target projection
    dot_product = np.dot(estimate, target)
    target_energy = np.dot(target, target) + eps
    scale = dot_product / target_energy
    s_target = scale * target
    
    # Interference (everything that's not target)
    e_interf = estimate - s_target
    
    # SIR
    signal_energy = np.dot(s_target, s_target) + eps
    interf_energy = np.dot(e_interf, e_interf) + eps
    sir = 10 * np.log10(signal_energy / interf_energy)
    
    return float(sir)


def evaluate_source_separation(
    estimates: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    metrics: list = ['si_sdr', 'sdr', 'sar', 'sir'],
) -> Dict[str, Dict[str, float]]:
    """Evaluate source separation performance.
    
    Args:
        estimates (Dict[str, torch.Tensor]): Estimated sources
        targets (Dict[str, torch.Tensor]): Target sources
        metrics (list): List of metrics to compute
        
    Returns:
        Dict[str, Dict[str, float]]: Metrics for each source
    """
    results = {}
    
    for source_name in estimates.keys():
        if source_name not in targets:
            continue
        
        estimate = estimates[source_name]
        target = targets[source_name]
        
        source_metrics = {}
        
        if 'si_sdr' in metrics:
            source_metrics['si_sdr'] = compute_si_sdr(estimate, target)
        
        if 'sdr' in metrics:
            source_metrics['sdr'] = compute_sdr(estimate, target)
        
        if 'sar' in metrics:
            source_metrics['sar'] = compute_sar(estimate, target)
        
        if 'sir' in metrics:
            source_metrics['sir'] = compute_sir(estimate, target)
        
        results[source_name] = source_metrics
    
    # Compute average across sources
    avg_metrics = {}
    for metric in metrics:
        values = [results[src][metric] for src in results.keys() if metric in results[src]]
        if values:
            avg_metrics[f'avg_{metric}'] = np.mean(values)
    
    results['average'] = avg_metrics
    
    return results


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print metrics in a formatted way.
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Metrics dictionary
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    for source_name, source_metrics in metrics.items():
        print(f"\n{source_name.upper()}:")
        for metric_name, value in source_metrics.items():
            print(f"  {metric_name}: {value:.3f} dB")
    
    print("="*60 + "\n")
