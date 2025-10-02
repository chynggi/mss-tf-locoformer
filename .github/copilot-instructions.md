# MSS-TF-Locoformer Copilot Instructions

## 프로젝트 개요
이 프로젝트는 TF-Locoformer 아키텍처를 음악 소스 분리(Music Source Separation, MSS)에 적용한 구현체입니다. 시간-주파수(Time-Frequency) 도메인에서 Transformer와 로컬 컨볼루션 모델링을 결합하여 음악의 각 소스(보컬, 드럼, 베이스, 기타 등)를 효과적으로 분리합니다.

## 기술 스택 및 주요 라이브러리
- **Deep Learning Framework**: PyTorch 2.0+
- **Audio Processing**: torchaudio, librosa, soundfile
- **Scientific Computing**: numpy, scipy
- **Data Handling**: pandas, h5py
- **Visualization**: matplotlib, tensorboard
- **Model Training**: pytorch-lightning (선택적)
- **Configuration**: hydra-core, omegaconf

## 프로젝트 구조
```
MSS-TF-Locoformer/
├── configs/           # 하이퍼파라미터 및 실험 설정
├── data/             # 데이터셋 관련 유틸리티
├── models/           # TF-Locoformer 모델 구현
├── training/         # 학습 관련 스크립트
├── evaluation/       # 평가 메트릭 및 스크립트
├── inference/        # 추론 및 음원 분리 실행
├── utils/            # 공통 유틸리티 함수
└── experiments/      # 실험 결과 및 체크포인트
```

## 코딩 스타일 가이드라인

### 1. 일반 원칙
- **가독성**: 명확하고 직관적인 변수명과 함수명 사용
- **모듈화**: 기능별로 명확히 분리된 모듈 구조
- **문서화**: docstring과 inline comment를 통한 충분한 설명
- **타입 힌팅**: Python 3.8+ type annotation 사용 권장

### 2. 네이밍 컨벤션
- **클래스**: PascalCase (예: `TFLocoformerModel`, `MSSDataset`)
- **함수/변수**: snake_case (예: `compute_loss`, `batch_size`)
- **상수**: UPPER_SNAKE_CASE (예: `SAMPLE_RATE`, `N_FFT`)
- **파일/모듈**: snake_case (예: `tf_locoformer.py`, `data_loader.py`)

### 3. 코드 구조 패턴
- **모델 구현**: nn.Module을 상속받은 클래스로 구현
- **데이터셋**: torch.utils.data.Dataset 상속
- **설정 관리**: dataclass 또는 OmegaConf 활용
- **로깅**: logging 모듈 또는 wandb 통합

## 주요 컴포넌트 및 클래스

### 1. 모델 아키텍처
```python
class TFLocoformerMSS(nn.Module):
    """MSS를 위한 TF-Locoformer 메인 모델"""
    # Encoder-Decoder 구조
    # Dual-path processing
    # Local convolution + Global attention
```

### 2. 데이터 처리
```python
class MSSDataset(Dataset):
    """음악 소스 분리 데이터셋 클래스"""
    # STFT 변환
    # 멀티 소스 타겟 준비
    # 데이터 증강 (선택적)
```

### 3. 손실 함수
```python
class MSSLoss(nn.Module):
    """MSS를 위한 복합 손실 함수"""
    # SI-SDR, L1, L2 손실 조합
    # 주파수 도메인 손실
    # Perceptual loss (선택적)
```

## 베스트 프랙티스

### 1. 오디오 처리
- **샘플링 레이트**: 44.1kHz 기본 사용
- **STFT 설정**: hop_length=1024, n_fft=2048 권장
- **정규화**: [-1, 1] 범위로 오디오 정규화
- **배치 처리**: 메모리 효율성을 위한 적절한 배치 크기 설정

### 2. 모델 학습
- **체크포인트**: 정기적인 모델 저장 및 복구 메커니즘
- **조기 종료**: validation loss 기반 early stopping
- **스케줄링**: learning rate scheduler 활용
- **그래디언트 클리핑**: 안정적인 학습을 위한 gradient clipping

### 3. 평가 메트릭
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
- **SAR/SIR**: Signal-to-Artifacts/Interference Ratio
- **PESQ/STOI**: 지각적 품질 측정 (보컬 분리시)

### 4. 코드 최적화
- **GPU 메모리**: torch.cuda.empty_cache() 적절한 사용
- **데이터 로딩**: num_workers 최적화
- **혼합 정밀도**: torch.cuda.amp 활용 권장

## 주석 및 문서화 가이드

### 함수 문서화 예시
```python
def separate_sources(mixture: torch.Tensor, 
                    model: nn.Module,
                    device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    음악 mixture에서 개별 소스들을 분리합니다.
    
    Args:
        mixture (torch.Tensor): 입력 음악 신호 [B, T]
        model (nn.Module): 학습된 TF-Locoformer 모델
        device (str): 연산 장치 ('cuda' or 'cpu')
    
    Returns:
        Dict[str, torch.Tensor]: 분리된 소스들 {'vocals', 'drums', 'bass', 'other'}
    """
```

### 설정 파일 구조
```yaml
# config/experiment/base.yaml
model:
  n_fft: 2048
  hop_length: 1024
  n_sources: 4
  
training:
  batch_size: 8
  learning_rate: 1e-4
  max_epochs: 200
  
data:
  dataset_path: "/path/to/musdb18"
  sample_rate: 44100
```

## 특별 고려사항
- **메모리 효율성**: 긴 오디오 시퀀스 처리시 청킹 기법 활용
- **실시간 처리**: 추론 최적화를 위한 모델 경량화 고려
- **다양한 장르**: 다양한 음악 장르에 대한 robust한 성능 추구
- **평가 일관성**: MUSDB18-HQ 등 표준 벤치마크 데이터셋 활용

이 지침을 바탕으로 일관되고 고품질의 MSS-TF-Locoformer 구현을 위한 코드를 작성해주시기 바랍니다.