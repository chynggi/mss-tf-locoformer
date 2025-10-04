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

### Option 1: 메모리 최적화 (추천)
```bash
python training/train.py --config configs/musdb18_memory_optimized.yaml
```

### Option 2: Small 설정 (안전)
```bash
python training/train.py --config configs/musdb18_small.yaml
```

### Option 3: 기존 체크포인트 이어서 (주의: 여전히 OOM 가능)
```bash
python training/train.py \
  --config configs/musdb18_memory_optimized.yaml \
  --resume experiments_xlarge/checkpoints/checkpoint_epoch*.pth
```

## 📈 예상 결과

**메모리 최적화 설정 적용 후**:
- VRAM 사용량: 96.6GB → **45-50GB** (약 50% 감소)
- Effective batch size: 16 → **8** (유지 가능)
- 학습 속도: 100% → **70-80%** (gradient accumulation 오버헤드)
- 모델 품질: 거의 동일 (약간 작은 모델이지만 충분한 용량)

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

여전히 OOM 발생 시:

1. `segment_length`를 더 줄이기: 441000 → 220500 (5초)
2. `n_layers`를 더 줄이기: 8 → 6 또는 4
3. `emb_dim`을 더 줄이기: 192 → 128
4. `gradient_checkpointing: true` 활성화

---

**즉시 실행을 권장합니다:**

```bash
# 현재 학습 중단 (Ctrl+C)
# 그리고:
python training/train.py --config configs/musdb18_memory_optimized.yaml
```
