#!/usr/bin/env python3
# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Inference script for music source separation.

This script loads a trained TF-Locoformer model and separates a music mixture
into individual sources (vocals, drums, bass, other).
"""

import argparse
import os
from pathlib import Path

import torch
import yaml

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.mss_tflocoformer import TFLocoformerMSS
from utils.audio import load_audio, save_audio
from utils.common import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Separate music sources using TF-Locoformer'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input audio file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./separated',
        help='Output directory for separated sources'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to model config file (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=44100,
        help='Sample rate for processing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path, config_path=None, device='cuda'):
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        config_path (str, optional): Path to config file
        device (str): Device to load model on
        
    Returns:
        TFLocoformerMSS: Loaded model
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config if provided
    if config_path is not None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
    else:
        # Use default config
        model_config = {}
    
    # Create model
    model = TFLocoformerMSS(**model_config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


def separate_audio(model, audio_path, output_dir, device='cuda', sample_rate=44100):
    """Separate audio file into sources.
    
    Args:
        model (TFLocoformerMSS): Trained model
        audio_path (str): Path to input audio file
        output_dir (str): Output directory
        device (str): Device
        sample_rate (int): Sample rate
    """
    print(f"\nProcessing: {audio_path}")
    
    # Load audio
    audio, sr = load_audio(audio_path, sample_rate=sample_rate, mono=False)
    print(f"Loaded audio: {audio.shape}, sample rate: {sr}")
    
    # Convert to mono if stereo (average channels)
    if audio.shape[0] > 1:
        audio_mono = audio.mean(dim=0, keepdim=True)
    else:
        audio_mono = audio
    
    # Add batch dimension and move to device
    audio_batch = audio_mono.unsqueeze(0).to(device)  # [1, 1, T]
    audio_batch = audio_batch.squeeze(1)  # [1, T]
    
    # Separate sources
    print("Separating sources...")
    with torch.no_grad():
        separated = model(audio_batch, return_time_domain=True)
    
    # Save separated sources
    os.makedirs(output_dir, exist_ok=True)
    input_name = Path(audio_path).stem
    
    for source_name, source_audio in separated.items():
        # Move to CPU
        source_audio = source_audio.squeeze(0).cpu()  # [T] or [C, T]
        
        # Convert mono to stereo by duplicating channel
        if source_audio.ndim == 1:
            source_audio = source_audio.unsqueeze(0).repeat(2, 1)
        elif source_audio.shape[0] == 1:
            source_audio = source_audio.repeat(2, 1)
        
        # Save
        output_path = os.path.join(output_dir, f"{input_name}_{source_name}.wav")
        save_audio(source_audio, output_path, sample_rate=sample_rate, normalize=True)
        print(f"Saved {source_name}: {output_path}")
    
    print(f"\nSeparation complete! Results saved to {output_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load model
    model = load_model(
        args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Separate audio
    separate_audio(
        model,
        args.input,
        args.output_dir,
        device=args.device,
        sample_rate=args.sample_rate
    )


if __name__ == '__main__':
    main()
