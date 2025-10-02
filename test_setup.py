#!/usr/bin/env python3
# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Quick test script to verify the MSS-TF-Locoformer setup.

This script checks if all required modules can be imported and
creates a dummy model to verify the installation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✓ torchaudio {torchaudio.__version__}")
    except ImportError as e:
        print(f"✗ torchaudio not found: {e}")
        return False
    
    try:
        from rotary_embedding_torch import RotaryEmbedding
        print("✓ rotary-embedding-torch")
    except ImportError as e:
        print(f"✗ rotary-embedding-torch not found: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML not found: {e}")
        return False
    
    print("\nTesting MSS-TF-Locoformer modules...")
    
    try:
        from models.mss_tflocoformer import TFLocoformerMSS
        print("✓ TFLocoformerMSS model")
    except ImportError as e:
        print(f"✗ TFLocoformerMSS model: {e}")
        return False
    
    try:
        from models.mss_loss import MSSLoss
        print("✓ MSSLoss")
    except ImportError as e:
        print(f"✗ MSSLoss: {e}")
        return False
    
    try:
        from data.mss_dataset import MUSDBDataset
        print("✓ MUSDBDataset")
    except ImportError as e:
        print(f"✗ MUSDBDataset: {e}")
        return False
    
    try:
        from evaluation.metrics import compute_si_sdr
        print("✓ Evaluation metrics")
    except ImportError as e:
        print(f"✗ Evaluation metrics: {e}")
        return False
    
    try:
        from utils.audio import load_audio
        from utils.common import set_seed
        print("✓ Utilities")
    except ImportError as e:
        print(f"✗ Utilities: {e}")
        return False
    
    return True


def test_model_creation():
    """Test creating a small model."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from models.mss_tflocoformer import TFLocoformerMSS
        
        # Create a small model
        model = TFLocoformerMSS(
            n_fft=1024,
            hop_length=512,
            n_sources=4,
            n_layers=2,
            emb_dim=32,
            n_heads=2,
            attention_dim=32,
            ffn_hidden_dim=64,
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully")
        print(f"  Parameters: {num_params:,}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 44100)  # 1 second of audio
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output sources: {list(output.keys())}")
        print(f"  Output shape (vocals): {output['vocals'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_function():
    """Test loss function."""
    print("\nTesting loss function...")
    
    try:
        import torch
        from models.mss_loss import MSSLoss, compute_si_sdr
        
        # Create loss function
        loss_fn = MSSLoss(loss_type='si_sdr')
        
        # Dummy predictions and targets
        predictions = {
            'vocals': torch.randn(2, 44100),
            'drums': torch.randn(2, 44100),
            'bass': torch.randn(2, 44100),
            'other': torch.randn(2, 44100),
        }
        
        targets = {
            'vocals': torch.randn(2, 44100),
            'drums': torch.randn(2, 44100),
            'bass': torch.randn(2, 44100),
            'other': torch.randn(2, 44100),
        }
        
        # Compute loss
        loss_dict = loss_fn(predictions, targets)
        
        print(f"✓ Loss function working")
        print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
        
        # Test SI-SDR metric
        si_sdr = compute_si_sdr(predictions['vocals'], targets['vocals'])
        print(f"✓ SI-SDR metric: {si_sdr.mean().item():.4f} dB")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MSS-TF-Locoformer Installation Test")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Some imports failed. Please install missing dependencies.")
    
    # Test model creation
    if not test_model_creation():
        all_passed = False
        print("\n⚠ Model creation failed.")
    
    # Test loss function
    if not test_loss_function():
        all_passed = False
        print("\n⚠ Loss function test failed.")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! MSS-TF-Locoformer is ready to use.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
