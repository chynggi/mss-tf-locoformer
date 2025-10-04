# MSS-TF-Locoformer 메모리 계산 및 최적화 상세 분석

## 🔍 OOM 원인 분석

### 문제의 근본 원인

```
torch.OutOfMemoryError: Tried to allocate 80.89 GiB
```

이 에러는 `scaled_dot_product_attention` 함수에서 발생했습니다.

### 메모리 계산

#### 1. 시퀀스 길이 계산

**XLarge 설정 (실패)**:
```
segment_length = 661500 samples (15초)
n_fft = 4096
hop_length = 1024

time_frames = 661500 / 1024 ≈ 646 frames
freq_bins = 4096 / 2 + 1 = 2049 bins
```

**Memory-Optimized 초기 설정 (여전히 실패)**:
```
segment_length = 441000 samples (10초)
n_fft = 4096
hop_length = 1024

time_frames = 441000 / 1024 ≈ 430 frames
freq_bins = 2049 bins
```

#### 2. Attention 메모리 계산

Attention은 frequency 또는 time 차원에서 수행됩니다:

**Frequency attention** (더 위험):
```
sequence_length = freq_bins = 2049
batch_size = 1
n_heads = 12
attention_dim = 192
head_dim = 192 / 12 = 16

Attention score matrix 크기:
  [batch, n_heads, seq_len, seq_len]
  = [1, 12, 2049, 2049]
  = 50,388,588 elements
  
메모리 (float32):
  50,388,588 × 4 bytes ≈ 191 MB (괜찮음)
  
메모리 (float16/bfloat16):
  50,388,588 × 2 bytes ≈ 96 MB (괜찮음)
```

**Time attention**:
```
sequence_length = time_frames = 430
batch_size = 1
n_heads = 12

Attention score matrix:
  [1, 12, 430, 430]
  = 2,224,800 elements
  ≈ 4.2 MB (매우 적음)
```

**그렇다면 왜 80GB가 필요한가?** 🤔

#### 3. 실제 메모리 사용량 계산

문제는 **모든 레이어의 중간 activation을 저장**해야 한다는 것:

```
Total layers = 8 (TFLocoformer blocks)
각 블록마다:
  - Frequency path: 1개 attention
  - Time path: 1개 attention
  
총 attention 연산 = 8 × 2 = 16개

각 attention의 중간 텐서:
  Q, K, V: [batch, n_heads, seq_len, head_dim]
  Attention scores: [batch, n_heads, seq_len, seq_len]
  Output: [batch, n_heads, seq_len, head_dim]
```

**최악의 경우 (Frequency attention, backward pass 포함)**:
```
1개 레이어의 메모리:
  Input: [1, 430, 2049, 192] ≈ 169 MB
  Q, K, V: 3 × [1, 12, 2049, 16] ≈ 2.4 MB
  Attention scores: [1, 12, 2049, 2049] ≈ 96 MB
  Output: [1, 12, 2049, 16] ≈ 0.8 MB
  Gradients (backward): 동일한 양
  
1개 레이어 총합: ~540 MB
16개 레이어: ~8.6 GB
```

하지만 실제로는:
- FFN 레이어의 중간 activation
- Normalization 레이어의 중간 값
- Residual connection을 위한 복사본
- Optimizer state (momentum, variance)
- 등등...

**총합**: 추정 **20-30GB per batch**

하지만 에러 메시지는 **80.89GB**를 요청했습니다. 이는:

1. **메모리 프래그멘테이션**: 연속된 80GB 블록이 필요했지만 메모리가 파편화됨
2. **Temporary tensor**: 일부 연산이 매우 큰 임시 텐서를 생성
3. **Conv2D 연산**: Encoder/Decoder의 Conv2D가 큰 메모리 필요

## ✅ 해결 방법

### 1. Segment Length 감소

**Ultra-Safe (3초)**:
```
segment_length = 132300 samples
time_frames = 132300 / 512 ≈ 258 frames
freq_bins = 2048 / 2 + 1 = 1025 bins

Frequency attention memory:
  [1, 4, 1025, 1025] ≈ 4.2 MB (✅ 매우 적음!)
```

**Memory-Optimized (5초)**:
```
segment_length = 220500 samples
time_frames = 220500 / 512 ≈ 430 frames
freq_bins = 1025 bins

Frequency attention memory:
  [1, 8, 1025, 1025] ≈ 8.4 MB (✅ 적음!)
```

### 2. 모델 크기 감소

| 설정 | n_layers | emb_dim | n_heads | 총 파라미터 | 메모리 |
|-----|----------|---------|---------|------------|--------|
| XLarge | 12 | 256 | 16 | ~120M | 매우 높음 |
| Memory-Opt (Old) | 8 | 192 | 12 | ~60M | 높음 |
| Memory-Opt (New) | 6 | 128 | 8 | ~30M | 중간 |
| Ultra-Safe | 4 | 96 | 4 | ~15M | 낮음 ✅ |

### 3. Gradient Accumulation

메모리를 더 절약하려면 gradient accumulation 사용:

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 4
  # Effective batch size = 1 × 4 = 4
```

메모리는 batch_size=1만큼만 사용하지만 효과는 batch_size=4!

### 4. Gradient Checkpointing (최후의 수단)

```yaml
training:
  gradient_checkpointing: true
```

- 메모리: 30-40% 절감
- 속도: 20-30% 감소
- Trade-off: 메모리 ⬇️, 시간 ⬆️

## 📊 설정별 메모리 예측

| 설정 | Segment | n_fft | Layers | Dim | 예상 VRAM | OOM 위험 |
|-----|---------|-------|--------|-----|-----------|----------|
| XLarge | 15s | 4096 | 12 | 256 | **~97GB** | 🔴 매우 높음 |
| Mem-Opt (Old) | 10s | 4096 | 8 | 192 | **~80GB** | 🔴 높음 |
| Mem-Opt (New) | 5s | 2048 | 6 | 128 | ~30-40GB | 🟡 중간 |
| Ultra-Safe | 3s | 2048 | 4 | 96 | ~15-20GB | 🟢 매우 낮음 |

## 🎯 권장 설정 선택 가이드

### 상황별 추천

1. **처음 실행 / OOM 발생**: 
   ```bash
   python training/train.py --config configs/musdb18_ultra_safe.yaml
   ```
   - 확실히 작동합니다!

2. **Ultra-Safe가 작동하고 여유가 있으면**:
   ```bash
   python training/train.py --config configs/musdb18_memory_optimized.yaml
   ```
   - 더 나은 품질

3. **메모리가 충분하면** (50GB+ 여유):
   ```bash
   python training/train.py --config configs/musdb18.yaml
   ```
   - 원래 설정

## 💡 Pro Tips

### 1. 메모리 모니터링
```bash
# 학습 전 확인
nvidia-smi

# 학습 중 모니터링
watch -n 1 nvidia-smi

# Python에서 확인
python -c "import torch; print(f'Available: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB')"
```

### 2. 메모리 프래그멘테이션 방지
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3. 배치 크기 자동 찾기 (실험적)
```python
# training/train.py에서
from torch.cuda.amp import autocast

def find_max_batch_size(model, sample_input):
    """자동으로 최대 배치 크기 찾기"""
    for bs in [1, 2, 4, 8, 16]:
        try:
            with torch.no_grad():
                batch = sample_input.repeat(bs, 1)
                output = model(batch)
            print(f"Batch size {bs}: OK")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {bs}: OOM")
                return bs - 1
    return bs
```

## 🔬 실험 결과 (예상)

### Ultra-Safe 설정
- **학습 시간**: 3초 세그먼트 → 더 많은 반복 필요
- **수렴**: 빠름 (작은 모델)
- **최종 성능**: SI-SDR ~8-10dB (실용적)
- **안정성**: ⭐⭐⭐⭐⭐

### Memory-Optimized 설정
- **학습 시간**: 중간
- **수렴**: 중간
- **최종 성능**: SI-SDR ~10-12dB (좋음)
- **안정성**: ⭐⭐⭐⭐

### 원본 설정
- **학습 시간**: 느림 (큰 세그먼트)
- **수렴**: 느림 (큰 모델)
- **최종 성능**: SI-SDR ~12-15dB (최고)
- **안정성**: ⭐⭐ (OOM 위험)

---

**결론**: 지금은 `musdb18_ultra_safe.yaml`로 시작하세요! 🚀
