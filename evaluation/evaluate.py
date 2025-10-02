#!/usr/bin/env python3
# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation script for MSS-TF-Locoformer.

Example usage:
    python evaluation/evaluate.py \
        --config configs/musdb18.yaml \
        --checkpoint experiments/checkpoints/best_model.pth \
        --output_dir evaluation_results
"""

import argparse
import os
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.mss_tflocoformer import TFLocoformerMSS
from data.mss_dataset import MUSDBDataset, collate_fn
from evaluation.metrics import evaluate_source_separation, print_metrics
from utils.common import set_seed, load_checkpoint
from utils.audio import save_audio


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MSS-TF-Locoformer')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save_audio',
        action='store_true',
        help='Save separated audio files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    return parser.parse_args()


def evaluate(model, dataloader, device, save_audio_flag=False, output_dir=None):
    """Evaluate model on dataset.
    
    Args:
        model: TF-Locoformer model
        dataloader: Test data loader
        device: Device
        save_audio_flag: Whether to save separated audio
        output_dir: Output directory for audio files
        
    Returns:
        dict: Evaluation results
    """
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            # Move data to device
            mixture = batch['mixture'].to(device)
            targets = {
                k: v.to(device) for k, v in batch.items()
                if k != 'mixture'
            }
            
            # Convert to mono if stereo
            if mixture.ndim == 3 and mixture.shape[1] == 2:
                mixture = mixture.mean(dim=1)
            
            # Forward pass
            predictions = model(mixture, return_time_domain=True)
            
            # Compute metrics
            batch_metrics = evaluate_source_separation(
                predictions, targets,
                metrics=['si_sdr', 'sdr', 'sar', 'sir']
            )
            all_metrics.append(batch_metrics)
            
            # Save audio if requested
            if save_audio_flag and output_dir:
                audio_dir = os.path.join(output_dir, f'track_{batch_idx:03d}')
                os.makedirs(audio_dir, exist_ok=True)
                
                for source_name, audio in predictions.items():
                    audio_path = os.path.join(audio_dir, f'{source_name}.wav')
                    save_audio(audio[0].cpu(), audio_path, sample_rate=44100)
    
    # Aggregate metrics
    aggregated_metrics = {}
    
    # Get all source names
    source_names = set()
    for metrics in all_metrics:
        source_names.update(metrics.keys())
    source_names.discard('average')
    
    # Average across all tracks
    for source_name in source_names:
        source_metrics = {}
        for metric_name in ['si_sdr', 'sdr', 'sar', 'sir']:
            values = [
                m[source_name][metric_name] 
                for m in all_metrics 
                if source_name in m and metric_name in m[source_name]
            ]
            if values:
                source_metrics[metric_name] = sum(values) / len(values)
        aggregated_metrics[source_name] = source_metrics
    
    # Compute overall average
    avg_metrics = {}
    for metric_name in ['si_sdr', 'sdr', 'sar', 'sir']:
        values = []
        for source_name in source_names:
            if source_name in aggregated_metrics and metric_name in aggregated_metrics[source_name]:
                values.append(aggregated_metrics[source_name][metric_name])
        if values:
            avg_metrics[f'avg_{metric_name}'] = sum(values) / len(values)
    
    aggregated_metrics['average'] = avg_metrics
    
    return aggregated_metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    print("Loading test dataset...")
    test_dataset = MUSDBDataset(
        root_dir=config['dataset']['root_dir'],
        subset='test',
        sample_rate=config['dataset']['sample_rate'],
        segment_length=None,  # Use full tracks
        sources=config['dataset']['sources'],
        augmentation=False,
        random_chunks=False,
    )
    
    print(f"Test dataset: {len(test_dataset)} tracks")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    # Create and load model
    print("Loading model...")
    model = TFLocoformerMSS(**config['model'])
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate(
        model, test_loader, device,
        save_audio_flag=args.save_audio,
        output_dir=args.output_dir
    )
    
    # Print results
    print_metrics(results)
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
