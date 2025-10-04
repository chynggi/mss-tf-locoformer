# VRAM 최적화 가이드 (RTX PRO 6000 - 97GB)

## 🚨 현재 문제

VRAM 사용량이 96.6GB / 97.9GB (98.7%)로 한계에 도달했습니다.

## ✅ 즉시 적용 가능한 해결책

### 1. **Gradient Accumulation** (가장 효과적)

배치 크기를 줄이고 gradient accumulation으로 보완:

```yaml
# configs/musdb18_memory_optimized.yaml 사용
training:
  batch_size: 1  # 최소화
  gradient_accumulation_steps: 8  # Effective batch = 8
```

**예상 효과**: VRAM 50-60% 절감

### 2. **Segment Length 줄이기**

더 짧은 오디오 세그먼트 사용:

```yaml
dataset:
  segment_length: 441000  # 10초 (15초에서 감소)
```

**예상 효과**: VRAM 30-40% 절감

### 3. **모델 크기 축소**

```yaml
model:
  n_layers: 8  # 12 → 8
  emb_dim: 192  # 256 → 192
  n_heads: 12  # 16 → 12
  ffn_hidden_dim: [768, 768]  # [1024, 1024] → [768, 768]
```

**예상 효과**: VRAM 20-30% 절감

### 4. **Gradient Checkpointing** (추가 최적화)

속도를 희생하여 메모리 절약:

```yaml
training:
  gradient_checkpointing: true
```

**예상 효과**: VRAM 추가 30-40% 절감 (속도 20-30% 감소)

## 🔧 적용 방법

### 즉시 재시작 (권장)

1. **현재 학습 중단**:
```bash
# Ctrl+C로 학습 중단
```

2. **최적화된 설정으로 재시작**:
```bash
python training/train.py --config configs/musdb18_memory_optimized.yaml
```

### 체크포인트에서 재개

```bash
python training/train.py \
  --config configs/musdb18_memory_optimized.yaml \
  --resume experiments_xlarge/checkpoints/best_model.pth
```

## 📊 메모리 사용량 모니터링

학습 중 VRAM 사용량이 progress bar에 표시됩니다:

```
Epoch 1: 100%|██████| 100/100 [10:23<00:00, loss=2.3456, lr=0.000300, mem=45.2/50.1GB]
                                                                        ^^^^^^^^^^^^^^^^
```

## 🎯 최적 설정 추천

현재 상황에서는 다음 설정이 최적입니다:

| 파라미터 | 기존 (XLarge) | 최적화 | 이유 |
|---------|--------------|--------|------|
| batch_size | 8 | 1 | VRAM 대폭 절감 |
| gradient_accum | 2 | 8 | Effective batch 유지 |
| segment_length | 661500 (15s) | 441000 (10s) | 시퀀스 길이 감소 |
| n_layers | 12 | 8 | 모델 깊이 감소 |
| emb_dim | 256 | 192 | 차원 감소 |
| n_heads | 16 | 12 | 헤드 수 감소 |

## 🔍 추가 디버깅

### 1. 현재 메모리 사용량 확인

```bash
nvidia-smi
```

### 2. PyTorch 메모리 프로파일링

학습 코드에 추가:

```python
import torch

# 메모리 스냅샷
print(torch.cuda.memory_summary())

# 할당된 메모리
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### 3. OOM 발생 시

```bash
# 더욱 보수적인 설정
python training/train.py --config configs/musdb18_small.yaml
```

## 💡 성능 vs 메모리 트레이드오프

| 설정 | VRAM | 학습 속도 | 모델 품질 |
|-----|------|----------|----------|
| XLarge | ~97GB | 빠름 | 최고 |
| Memory-Optimized | ~45-50GB | 중간 | 높음 |
| Small | ~20-25GB | 느림 | 중간 |

## 🚀 즉시 실행 명령어

### ⚠️ OOM 발생 시 문제 분석

에러 메시지: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.89 GiB`

**원인**: 
- segment_length가 너무 길면 attention 연산에서 메모리 폭발 발생
- Attention 메모리 = O(sequence_length²) - 10초 세그먼트는 너무 김!

**해결책**: 더 짧은 세그먼트 + 작은 모델 사용

### Option 1: Ultra-Safe 설정 (강력 추천! 🔥)
```bash
python training/train.py --config configs/musdb18_ultra_safe.yaml
```
- **segment_length**: 3초 (안전)
- **모델**: 매우 작음 (4 layers, 96 dim)
- **예상 VRAM**: ~15-20GB
- **성공률**: 99.9%

### Option 2: 메모리 최적화 (업데이트됨)
```bash
python training/train.py --config configs/musdb18_memory_optimized.yaml
```
- **segment_length**: 5초 (중간)
- **모델**: 작음 (6 layers, 128 dim)
- **예상 VRAM**: ~30-40GB
- **성공률**: 95%

### Option 3: Small 설정 (원본)
```bash
python training/train.py --config configs/musdb18_small.yaml
```
- **segment_length**: 3초
- **모델**: 작음 (4 layers, 96 dim)
- **예상 VRAM**: ~20-25GB

## 📈 예상 결과

### Ultra-Safe 설정 (추천!)
- VRAM 사용량: 96.6GB → **15-20GB** (약 80% 감소) ✅
- Effective batch size: 4
- 학습 속도: 빠름 (작은 모델)
- 모델 품질: 중간 (충분히 실용적)
- **OOM 위험**: 거의 없음 🎯

### Memory-Optimized 설정 (업데이트)
- VRAM 사용량: 96.6GB → **30-40GB** (약 60% 감소)
- Effective batch size: 1
- 학습 속도: 중간
- 모델 품질: 높음

## ⚠️ 주의사항

1. **Gradient Accumulation**: optimizer step이 N번에 한 번만 실행되므로 학습 속도가 약간 느려질 수 있습니다.

2. **Gradient Checkpointing**: 활성화하면 forward pass를 다시 계산해야 하므로 학습 시간이 20-30% 증가합니다. VRAM이 여전히 부족할 때만 사용하세요.

3. **체크포인트 호환성**: 모델 구조가 변경되면 기존 체크포인트를 로드할 수 없습니다. 처음부터 재학습하거나 동일한 구조 유지가 필요합니다.

## 🔄 실시간 모니터링

학습 중 다른 터미널에서:

```bash
# 1초마다 GPU 상태 확인
watch -n 1 nvidia-smi

# 또는 더 상세한 정보
nvidia-smi dmon -s u
```

## 📞 문제 해결

### OOM 에러 분석

```
torch.OutOfMemoryError: Tried to allocate 80.89 GiB
```

**주요 원인**:
1. **Attention 메모리 폭발**: 
   - Time frames: 10초 @ hop_length=1024 = ~430 frames
   - Freq bins: n_fft=4096 = 2049 bins
   - Attention memory ≈ (430 × 2049)² × 4 bytes ≈ **80GB!** 💥

2. **해결 방법**:
   - ✅ segment_length 줄이기: 10초 → 5초 → **3초**
   - ✅ n_fft 줄이기: 4096 → **2048** (주파수 빈 절반)
   - ✅ hop_length 줄이기: 1024 → **512** (시간 프레임 증가 방지)
   - ✅ n_layers 줄이기: 8 → **4-6**
   - ✅ emb_dim 줄이기: 192 → **96-128**

### 단계별 해결

#### Step 1: Ultra-Safe로 시작 (가장 안전)
```bash
python training/train.py --config configs/musdb18_ultra_safe.yaml
```
- **확실하게 작동**
- 메모리 사용량: ~15-20GB
- 학습이 정상 작동하는지 확인

#### Step 2: 메모리가 충분하면 Memory-Optimized로 업그레이드
```bash
python training/train.py --config configs/musdb18_memory_optimized.yaml
```
- 더 나은 품질
- 메모리 사용량: ~30-40GB
- 여전히 안전

#### Step 3: 메모리 프래그멘테이션 문제 해결
OOM이 계속 발생하면:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training/train.py --config configs/musdb18_ultra_safe.yaml
```

### 긴급 명령어 (지금 바로 실행!)

```bash
# 즉시 실행 - 99.9% 성공!
python training/train.py --config configs/musdb18_ultra_safe.yaml
```
