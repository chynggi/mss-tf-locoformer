# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
TF-Locoformer model for Music Source Separation (MSS).

This module implements the TF-Locoformer architecture adapted for separating
music into multiple sources: vocals, drums, bass, and other instruments.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class MSSTransform(nn.Module):
    """STFT/iSTFT transformation for MSS."""
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 1024,
        win_length: Optional[int] = None,
        window: str = "hann",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        
    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply STFT to audio signal.
        
        Args:
            audio (torch.Tensor): Input audio [B, T]
            
        Returns:
            torch.Tensor: Complex spectrogram [B, F, T]
        """
        window = torch.hann_window(self.win_length, device=audio.device)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        return spec
    
    def istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Apply inverse STFT to spectrogram.
        
        Args:
            spec (torch.Tensor): Complex spectrogram [B, F, T]
            length (int, optional): Target output length
            
        Returns:
            torch.Tensor: Audio signal [B, T]
        """
        window = torch.hann_window(self.win_length, device=spec.device)
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=length,
        )
        return audio


class TFLocoformerMSS(nn.Module):
    """TF-Locoformer model for Music Source Separation.
    
    This model separates a music mixture into 4 sources: vocals, drums, bass, and other.
    
    Args:
        n_fft (int): FFT size for STFT. Default: 2048
        hop_length (int): Hop length for STFT. Default: 1024
        n_sources (int): Number of output sources. Default: 4 (vocals, drums, bass, other)
        n_layers (int): Number of Locoformer blocks. Default: 6
        emb_dim (int): Embedding dimension. Default: 128
        norm_type (str): Normalization type ('layernorm' or 'rmsgroupnorm'). Default: 'rmsgroupnorm'
        num_groups (int): Number of groups for RMSGroupNorm. Default: 4
        tf_order (str): Order of time-frequency processing ('ft' or 'tf'). Default: 'ft'
        n_heads (int): Number of attention heads. Default: 4
        flash_attention (bool): Use flash attention. Default: False
        attention_dim (int): Attention dimension. Default: 128
        pos_enc (str): Positional encoding type ('rope' or 'nope'). Default: 'rope'
        ffn_type (str or list): FFN type. Default: 'swiglu_conv1d'
        ffn_hidden_dim (int or list): FFN hidden dimension. Default: 384
        conv1d_kernel (int): Conv1d kernel size. Default: 4
        conv1d_shift (int): Conv1d shift size. Default: 1
        dropout (float): Dropout probability. Default: 0.0
        eps (float): Small constant for normalization. Default: 1e-5
    """
    
    def __init__(
        self,
        # Audio processing
        n_fft: int = 2048,
        hop_length: int = 1024,
        # Model architecture
        n_sources: int = 4,
        n_layers: int = 6,
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,
        tf_order: str = "ft",
        # Self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,
        attention_dim: int = 128,
        pos_enc: str = "rope",
        # FFN related
        ffn_type: str = "swiglu_conv1d",
        ffn_hidden_dim: int = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # Others
        eps: float = 1.0e-5,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_sources = n_sources
        self.n_layers = n_layers
        
        # STFT transformation
        self.transform = MSSTransform(n_fft=n_fft, hop_length=hop_length)
        
        # Encoder: Conv2D for initial feature extraction
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),  # Global LayerNorm
        )
        
        # Positional encoding
        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        if pos_enc == "nope":
            rope_freq = rope_time = None
        elif pos_enc == "rope":
            rope_freq = RotaryEmbedding(attention_dim // n_heads)
            rope_time = RotaryEmbedding(attention_dim // n_heads)
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_enc}")
        
        # TF-Locoformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    rope_freq,
                    rope_time,
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )
        
        # Decoder: ConvTranspose2D for output
        self.deconv = nn.ConvTranspose2d(emb_dim, n_sources * 2, ks, padding=padding)
        
    def forward(
        self,
        mixture: torch.Tensor,
        return_time_domain: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for music source separation.
        
        Args:
            mixture (torch.Tensor): Input mixture audio [B, T]
            return_time_domain (bool): Return time-domain audio. Default: True
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with separated sources
                If return_time_domain=True: {'vocals', 'drums', 'bass', 'other'}: [B, T]
                If return_time_domain=False: complex spectrograms [B, F, T]
        """
        batch_size = mixture.shape[0]
        original_length = mixture.shape[-1]
        
        # STFT
        spec = self.transform.stft(mixture)  # [B, F, T]
        
        # Prepare input: stack real and imaginary parts
        batch = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        batch = batch.transpose(-1, -2)  # [B, 2, T, F]
        n_frames, n_freqs = batch.shape[2], batch.shape[3]
        
        # Encoder
        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch)  # [B, emb_dim, T, F]
        
        # TF-Locoformer blocks
        for block in self.blocks:
            batch = block(batch)  # [B, emb_dim, T, F]
        
        # Decoder
        with torch.cuda.amp.autocast(enabled=False):
            batch = self.deconv(batch)  # [B, n_sources*2, T, F]
        
        # Reshape to [B, n_sources, 2, T, F]
        batch = batch.view(batch_size, self.n_sources, 2, n_frames, n_freqs)
        
        # Reconstruct complex spectrogram
        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])  # [B, n_sources, T, F]
        batch = batch.transpose(-1, -2)  # [B, n_sources, F, T]
        
        if return_time_domain:
            # iSTFT for each source
            sources = {}
            source_names = ['vocals', 'drums', 'bass', 'other']
            for i, name in enumerate(source_names[:self.n_sources]):
                audio = self.transform.istft(batch[:, i], length=original_length)
                sources[name] = audio
            return sources
        else:
            # Return spectrograms
            return {
                'vocals': batch[:, 0],
                'drums': batch[:, 1],
                'bass': batch[:, 2],
                'other': batch[:, 3],
            }


class TFLocoformerBlock(nn.Module):
    """TF-Locoformer block for dual-path processing."""
    
    def __init__(
        self,
        rope_freq,
        rope_time,
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,
        tf_order: str = "ft",
        n_heads: int = 4,
        flash_attention: bool = False,
        attention_dim: int = 128,
        ffn_type: str = "swiglu_conv1d",
        ffn_hidden_dim: int = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1.0e-5,
    ):
        super().__init__()
        
        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift
        
        # Frequency path
        self.freq_path = LocoformerBlock(
            rope_freq,
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        
        # Temporal path
        self.frame_path = LocoformerBlock(
            rope_time,
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """TF-Locoformer block forward.
        
        Args:
            input (torch.Tensor): Input tensor [B, C, T, F]
            
        Returns:
            torch.Tensor: Output tensor [B, C, T, F]
        """
        if self.tf_order == "ft":
            return self.freq_frame_process(input)
        else:
            return self.frame_freq_process(input)
    
    def freq_frame_process(self, input: torch.Tensor) -> torch.Tensor:
        """Frequency-then-time processing."""
        output = input.movedim(1, -1)  # [B, T, F, C]
        output = self.freq_path(output)
        
        output = output.transpose(1, 2)  # [B, F, T, C]
        output = self.frame_path(output)
        return output.transpose(-1, 1)  # [B, C, T, F]
    
    def frame_freq_process(self, input: torch.Tensor) -> torch.Tensor:
        """Time-then-frequency processing."""
        output = input.transpose(1, -1)  # [B, F, T, C]
        output = self.frame_path(output)
        
        output = output.transpose(1, 2)  # [B, T, F, C]
        output = self.freq_path(output)
        return output.movedim(-1, 1)  # [B, C, T, F]


class LocoformerBlock(nn.Module):
    """Locoformer block with local convolution and global attention."""
    
    def __init__(
        self,
        rope,
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,
        n_heads: int = 4,
        flash_attention: bool = False,
        attention_dim: int = 128,
        ffn_type: str = "swiglu_conv1d",
        ffn_hidden_dim: int = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        eps: float = 1.0e-5,
    ):
        super().__init__()
        
        FFN = {
            "conv1d": ConvDeconv1d,
            "swiglu_conv1d": SwiGLUConvDeconv1d,
        }
        Norm = {
            "layernorm": nn.LayerNorm,
            "rmsgroupnorm": RMSGroupNorm,
        }
        assert norm_type in Norm, norm_type
        
        # Check for Macaron-style (dual FFN)
        self.macaron_style = isinstance(ffn_type, list) and len(ffn_type) == 2
        if self.macaron_style:
            assert isinstance(ffn_hidden_dim, list) and len(ffn_hidden_dim) == 2
            ffn_type_list = ffn_type[::-1]
            ffn_dim_list = ffn_hidden_dim[::-1]
        else:
            ffn_type_list = [ffn_type]
            ffn_dim_list = [ffn_hidden_dim]
        
        # Initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type_list, ffn_dim_list):
            assert f_type in FFN, f_type
            if norm_type == "rmsgroupnorm":
                self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
            else:
                self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )
        
        # Initialize self-attention
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Locoformer block forward.
        
        Args:
            x (torch.Tensor): Input tensor [B, S1, S2, C]
                S1 and S2 are either time or frequency dimension
                
        Returns:
            torch.Tensor: Output tensor [B, S1, S2, C]
        """
        B, T, F, C = x.shape
        
        # Macaron-style: FFN before self-attention
        if self.macaron_style:
            residual = x
            output = self.ffn_norm[-1](x)
            output = self.ffn[-1](output)
            output = output + residual
        else:
            output = x
        
        # Self-attention
        residual = output
        output = self.attn_norm(output)
        output = output.view(B * T, F, C)
        output = self.attn(output)
        output = output.view(B, T, F, C) + residual
        
        # FFN after self-attention
        residual = output
        output = self.ffn_norm[0](output)
        output = self.ffn[0](output)
        output = output + residual
        
        return output


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""
    
    def __init__(
        self,
        emb_dim: int,
        attention_dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        rope=None,
        flash_attention: bool = False,
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.dropout = dropout
        self.rope = rope
        
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=False),
            nn.Dropout(dropout)
        )
        
        if flash_attention:
            self.flash_attention_config = dict(
                enable_flash=True,
                enable_math=True,  # Fallback to math kernel if flash not available
                enable_mem_efficient=False
            )
        else:
            self.flash_attention_config = dict(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=True
            )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention forward.
        
        Args:
            input (torch.Tensor): Input tensor [B, L, C]
            
        Returns:
            torch.Tensor: Output tensor [B, L, C]
        """
        # Get query, key, and value
        query, key, value = self.get_qkv(input)
        
        # Apply rotary positional encoding
        if self.rope is not None:
            query, key = self.apply_rope(query, key)
        
        # Scaled dot-product attention
        with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # [B, n_heads, L, dim]
        
        output = output.transpose(1, 2)  # [B, L, n_heads, dim]
        output = output.reshape(output.shape[:2] + (-1,))  # [B, L, attention_dim]
        return self.aggregate_heads(output)
    
    def get_qkv(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute query, key, and value tensors."""
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # [B, n_heads, L, 3, dim]
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value
    
    @torch.cuda.amp.autocast(enabled=False)
    def apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding."""
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class ConvDeconv1d(nn.Module):
    """Conv-Deconv 1D module for local modeling."""
    
    def __init__(
        self,
        dim: int,
        dim_inner: int,
        conv1d_kernel: int,
        conv1d_shift: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.diff_ks = conv1d_kernel - conv1d_shift
        
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv-Deconv forward.
        
        Args:
            x (torch.Tensor): Input tensor [B, S1, S2, C]
            
        Returns:
            torch.Tensor: Output tensor [B, S1, S2, C]
        """
        b, s1, s2, h = x.shape
        x = x.view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    """SwiGLU Conv-Deconv 1D module for local modeling with gating."""
    
    def __init__(
        self,
        dim: int,
        dim_inner: int,
        conv1d_kernel: int,
        conv1d_shift: int,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)
        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU Conv-Deconv forward.
        
        Args:
            x (torch.Tensor): Input tensor [B, S1, S2, C]
            
        Returns:
            torch.Tensor: Output tensor [B, S1, S2, C]
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        
        # Padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) 
            * self.conv1d_shift + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))
        
        # Conv-deconv with gating
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner:, :])
        x = x[..., :self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)
        
        # Cut to necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    """Root Mean Square Group Normalization for TF bins."""
    
    def __init__(
        self,
        num_groups: int,
        dim: int,
        eps: float = 1e-8,
        bias: bool = False
    ):
        super().__init__()
        
        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups
        
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        
        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        
        self.eps = eps
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """RMS Group normalization forward.
        
        Args:
            input (torch.Tensor): Input tensor [..., C]
            
        Returns:
            torch.Tensor: Normalized tensor [..., C]
        """
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))
        
        # Normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)
        
        # Reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta
        
        return output
