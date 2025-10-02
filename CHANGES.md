# MSS-TF-Locoformer 프로젝트 변형 요약

## 개요

기존 TF-Locoformer (Speech Separation/Enhancement) 프로젝트를 음악 소스 분리(Music Source Separation, MSS)용으로 변형했습니다.

## 주요 변경사항

### 1. 프로젝트 구조 재구성

새로운 디렉토리 구조 생성:
```
mss-tf-locoformer/
├── configs/              # 설정 파일
├── data/                 # 데이터셋 로더
├── models/               # 모델 아키텍처
├── training/             # 학습 스크립트
├── evaluation/           # 평가 메트릭
├── inference/            # 추론 스크립트
└── utils/                # 유틸리티 함수
```

### 2. 새로 생성된 핵심 파일

#### 모델 (models/)
- **mss_tflocoformer.py**: MSS용 TF-Locoformer 구현
  - 4개 소스(vocals, drums, bass, other) 동시 분리
  - STFT/iSTFT 통합
  - 44.1kHz 샘플링 레이트, 2048 FFT 크기 지원

- **mss_loss.py**: 복합 손실 함수
  - SI-SDR Loss
  - L1/L2 Loss  
  - Spectral Loss
  - Multi-resolution STFT Loss

#### 데이터 (data/)
- **mss_dataset.py**: MUSDB18/HQ 데이터셋 로더
  - PyTorch Dataset 구현
  - 데이터 증강 (랜덤 gain, 채널 스와핑)
  - 세그먼트 추출
  - Collate 함수

#### 학습 (training/)
- **train.py**: 메인 학습 스크립트
  - AdamW optimizer
  - ReduceLROnPlateau scheduler
  - Gradient clipping
  - Checkpoint 저장
  - TensorBoard 로깅

#### 평가 (evaluation/)
- **metrics.py**: 평가 메트릭
  - SI-SDR (Scale-Invariant SDR)
  - SDR (Signal-to-Distortion Ratio)
  - SAR (Signal-to-Artifacts Ratio)
  - SIR (Signal-to-Interference Ratio)

- **evaluate.py**: 평가 스크립트
  - 테스트셋 평가
  - 메트릭 계산 및 저장
  - 분리된 오디오 저장 옵션

#### 추론 (inference/)
- **separate.py**: 음원 분리 스크립트
  - 단일 오디오 파일 처리
  - 4개 소스로 분리
  - WAV 형식 저장

#### 유틸리티 (utils/)
- **audio.py**: 오디오 처리 함수
  - 로드/저장
  - 정규화
  - STFT 계산
  - Gain 적용

- **common.py**: 공통 유틸리티
  - 랜덤 시드 설정
  - 체크포인트 관리
  - 평균 측정기
  - 시간 포맷팅

#### 설정 (configs/)
- **musdb18.yaml**: 기본 모델 설정
  - 6 layers, 128 embedding dim
  - 2048 FFT, 1024 hop length
  - Combined loss (SI-SDR + L1 + Spectral)

- **musdb18_small.yaml**: 작은 모델 설정
  - 4 layers, 64 embedding dim
  - 1024 FFT, 512 hop length
  - SI-SDR loss only

### 3. 수정된 파일

#### README.md
- MSS 프로젝트 소개로 전면 재작성
- 한국어 섹션 추가
- 설치 및 사용 가이드
- 문제 해결 섹션
- 사용 예시 코드

#### requirements.txt
- MSS에 필요한 의존성 추가
- musdb, museval 패키지 추가
- 버전 명시

#### .github/copilot-instructions.md
- MSS 프로젝트에 맞는 가이드라인 설정
- 코딩 스타일 및 베스트 프랙티스

### 4. 주요 기능

#### 모델 아키텍처
- Dual-path processing (시간축/주파수축 독립 처리)
- Multi-head self-attention with RoPE
- SwiGLU Feed-forward network
- RMS Group Normalization

#### 손실 함수
- Scale-invariant loss for music
- Time-domain + Frequency-domain
- 가중치 조절 가능

#### 데이터 증강
- 소스별 랜덤 gain (0.7~1.3)
- 채널 스와핑
- Polarity inversion
- 랜덤 청크 추출

#### 학습 기능
- Mixed precision training (AMP) 지원
- Gradient clipping
- Learning rate scheduling
- Early stopping
- Checkpoint 관리

#### 평가 메트릭
- SI-SDR: 음원 분리 품질
- SDR: 전체 왜곡
- SAR: 아티팩트
- SIR: 간섭

### 5. 사용 방법

#### 학습
```bash
python training/train.py --config configs/musdb18.yaml
```

#### 추론
```bash
python inference/separate.py \
    --input music.wav \
    --output_dir ./separated \
    --checkpoint best_model.pth
```

#### 평가
```bash
python evaluation/evaluate.py \
    --config configs/musdb18.yaml \
    --checkpoint best_model.pth
```

#### 설정 테스트
```bash
python test_setup.py
```

### 6. 기술적 세부사항

#### STFT 파라미터
- **n_fft**: 2048 (기본) / 1024 (작은 모델)
- **hop_length**: 1024 (기본) / 512 (작은 모델)
- **window**: Hann window
- **sample_rate**: 44100 Hz

#### 모델 크기
- **기본 모델**: ~2-3M parameters
- **작은 모델**: ~500K-1M parameters

#### 메모리 요구사항
- **학습**: 8-16GB GPU RAM (batch_size=4)
- **추론**: 4-8GB GPU RAM

### 7. 원본 프로젝트와의 호환성

원본 TF-Locoformer 코드는 다음 위치에 보존:
- `espnet2/enh/separator/tflocoformer_separator.py` (ESPnet 호환)
- `standalone/tflocoformer_separator.py` (독립 실행형)
- `egs2/*/enh1/` (ESPnet 레시피)

이들은 Speech Separation/Enhancement에 계속 사용 가능합니다.

### 8. 다음 단계

#### 즉시 가능한 작업
1. MUSDB18-HQ 데이터셋 다운로드
2. 의존성 설치 (`pip install -r requirements.txt`)
3. 설정 파일 경로 수정
4. 학습 시작

#### 향후 개선 사항
- Multi-GPU 분산 학습
- Pre-trained weights 제공
- Real-time inference 최적화
- Web demo interface
- Additional datasets 지원

### 9. 참고사항

#### 데이터셋
- MUSDB18-HQ: https://zenodo.org/record/3338373
- 학습: 100 tracks, 테스트: 50 tracks
- 전체 크기: ~약 30GB

#### 성능 목표
- SI-SDR: > 6 dB (MUSDB18 테스트셋)
- SDR: > 8 dB
- 학습 시간: ~2-4일 (4 GPU, RTX 2080Ti)

#### 코딩 규칙
- PEP 8 스타일 가이드 준수
- Type hints 사용
- Docstring (Google style)
- 함수/클래스명: snake_case / PascalCase

## 결론

기존 TF-Locoformer를 음악 소스 분리에 성공적으로 적용했습니다. 
모든 핵심 컴포넌트(모델, 데이터, 학습, 평가, 추론)가 구현되어 즉시 사용 가능합니다.
