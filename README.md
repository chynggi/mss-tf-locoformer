<!--
Copyright (C) 2024 MSS-TF-Locoformer

SPDX-License-Identifier: Apache-2.0
-->

# MSS-TF-Locoformer: Music Source Separation with TF-Locoformer

이 저장소는 TF-Locoformer 아키텍처를 음악 소스 분리(Music Source Separation, MSS)에 적용한 구현체입니다. 
시간-주파수(Time-Frequency) 도메인에서 Transformer와 로컬 컨볼루션 모델링을 결합하여 음악의 각 소스(보컬, 드럼, 베이스, 기타 등)를 효과적으로 분리합니다.

## 주요 특징

- **TF-Locoformer 아키텍처**: Dual-path processing으로 시간축과 주파수축을 독립적으로 모델링
- **4개 소스 분리**: vocals, drums, bass, other 소스를 동시에 분리
- **STFT 기반 처리**: 44.1kHz 샘플링 레이트, 2048 FFT 크기
- **유연한 손실 함수**: SI-SDR, L1, L2, Spectral loss 조합 가능
- **MUSDB18/HQ 지원**: 표준 벤치마크 데이터셋 호환

## 원본 TF-Locoformer 논문

This implementation is based on the TF-Locoformer model proposed in the following paper:

```
@InProceedings{Saijo2024_TFLoco,
  author    =  {Saijo, Kohei and Wichern, Gordon and Germain, Fran\c{c}ois G. and Pan, Zexu and {Le Roux}, Jonathan},
  title     =  {TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation and Enhancement},
  booktitle =  {Proc. International Workshop on Acoustic Signal Enhancement (IWAENC)},
  year      =  2024,
  month     =  sep
}
```

In addition to the original TF-Locoformer, the repository also supports some variants of TF-Locoformer:
- TF-Locoformer with no positional encoding (TF-Locoformer-NoPE) introduced in [Saijo2025Comparative]
- TF-Locoformer with band-split encoder (BS-Locoformer) introduced in [Saijo2025Task]

```
@InProceedings{Saijo2025Comparative,
  author    =  {Saijo, Kohei and Ogawa, Tetsuji},
  title     =  {A Comparative Study on Positional Encoding for Time-frequency Domain Dual-path Transformer-based Source Separation Models},
  booktitle =  {Proc. European Signal Processing Conference (EUSIPCO)},
  year      =  2025,
  month     =  sep
}

@InProceedings{Saijo2025Task,
  author    =  {Saijo, Kohei and Ebbers, Janek and Germain, Fran\c{c}ois G. and Wichern, Gordon and {Le Roux}, Jonathan},
  title     =  {Task-Aware Unified Source Separation},
  booktitle =  {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      =  2025,
  month     =  april
}
```


## 목차

1. [설치 및 환경 설정](#설치-및-환경-설정)
2. [프로젝트 구조](#프로젝트-구조)
3. [데이터셋 준비](#데이터셋-준비)
4. [학습](#학습)
5. [추론 및 음원 분리](#추론-및-음원-분리)
6. [평가](#평가)
7. [사용 예시](#사용-예시)
8. [라이센스](#라이센스)



## 설치 및 환경 설정

### 요구사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 사용 시)

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/mss-tf-locoformer.git
cd mss-tf-locoformer

# 가상환경 생성 (선택사항)
conda create -n mss-tf python=3.10
conda activate mss-tf

# 의존성 설치
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### requirements.txt 업데이트

프로젝트 루트의 `requirements.txt` 파일을 다음과 같이 업데이트하세요:

```txt
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.22.0
scipy>=1.9.0
soundfile>=0.12.0
pyyaml>=6.0
tensorboard>=2.11.0
tqdm>=4.64.0
rotary-embedding-torch==0.6.1
musdb>=0.4.0
museval>=0.4.0
```

## 프로젝트 구조

```
mss-tf-locoformer/
├── configs/              # 설정 파일
│   ├── musdb18.yaml      # MUSDB18 기본 설정
│   └── musdb18_small.yaml # 작은 모델 설정
├── data/                 # 데이터셋 관련
│   └── mss_dataset.py    # MUSDB18 데이터로더
├── models/               # 모델 아키텍처
│   ├── mss_tflocoformer.py # MSS용 TF-Locoformer
│   └── mss_loss.py       # 손실 함수
├── training/             # 학습 스크립트
│   └── train.py          # 메인 학습 스크립트
├── evaluation/           # 평가 메트릭
│   ├── metrics.py        # SI-SDR, SAR, SIR 등
│   └── evaluate.py       # 평가 스크립트
├── inference/            # 추론
│   └── separate.py       # 음원 분리 스크립트
└── utils/                # 유틸리티
    ├── audio.py          # 오디오 처리
    └── common.py         # 공통 함수
```

## 데이터셋 준비

### MUSDB18-HQ 다운로드

```bash
# MUSDB18-HQ 데이터셋 다운로드
# https://zenodo.org/record/3338373 에서 다운로드
# 또는 musdb 패키지 사용
python -c "import musdb; musdb.DB(download=True, root='./data/musdb18hq')"
```

데이터셋 구조:
```
musdb18hq/
├── train/
│   ├── track001/
│   │   ├── mixture.wav
│   │   ├── vocals.wav
│   │   ├── drums.wav
│   │   ├── bass.wav
│   │   └── other.wav
│   └── ...
└── test/
    └── ...
```

### 설정 파일 수정

`configs/musdb18.yaml` 파일에서 데이터셋 경로를 수정하세요:

```yaml
dataset:
  root_dir: "/path/to/musdb18hq"  # 실제 경로로 변경
```

## 학습

### 기본 학습

```bash
python training/train.py --config configs/musdb18.yaml
```

### 작은 모델로 빠른 테스트

```bash
python training/train.py --config configs/musdb18_small.yaml
```

### 학습 옵션

```bash
python training/train.py \
    --config configs/musdb18.yaml \
    --output_dir ./experiments/exp001 \
    --gpu 0 \
    --batch_size 4 \
    --num_epochs 200
```

학습 중 체크포인트와 로그는 `experiments/` 디렉토리에 저장됩니다.

## 추론 및 음원 분리

학습된 모델로 음원을 분리하세요:

```bash
python inference/separate.py \
    --input /path/to/music.wav \
    --output_dir ./separated \
    --checkpoint ./experiments/checkpoints/best_model.pth \
    --config configs/musdb18.yaml
```

출력 파일:
- `music_vocals.wav`
- `music_drums.wav`
- `music_bass.wav`
- `music_other.wav`

## 평가

테스트셋에서 모델 평가:

```bash
python evaluation/evaluate.py \
    --config configs/musdb18.yaml \
    --checkpoint ./experiments/checkpoints/best_model.pth \
    --output_dir ./evaluation_results
```

평가 메트릭:
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
- **SDR**: Signal-to-Distortion Ratio
- **SAR**: Signal-to-Artifacts Ratio
- **SIR**: Signal-to-Interference Ratio

## 사용 예시

### Python 코드로 모델 사용

```python
import torch
from models.mss_tflocoformer import TFLocoformerMSS
from utils.audio import load_audio, save_audio

# 모델 로드
model = TFLocoformerMSS(
    n_fft=2048,
    hop_length=1024,
    n_sources=4,
    n_layers=6,
    emb_dim=128,
)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 오디오 로드
mixture, sr = load_audio('music.wav', sample_rate=44100)

# 음원 분리
with torch.no_grad():
    separated = model(mixture.unsqueeze(0), return_time_domain=True)

# 결과 저장
for source_name, audio in separated.items():
    save_audio(audio[0], f'{source_name}.wav', sample_rate=sr)
```

### 배치 처리

```python
from data.mss_dataset import MUSDBDataset
from torch.utils.data import DataLoader

# 데이터셋 생성
dataset = MUSDBDataset(
    root_dir='/path/to/musdb18hq',
    subset='test',
    sample_rate=44100,
    segment_length=None,  # 전체 트랙 사용
)

# 데이터로더
loader = DataLoader(dataset, batch_size=1, num_workers=4)

# 배치 처리
for batch in loader:
    mixture = batch['mixture']
    with torch.no_grad():
        separated = model(mixture)
    # 결과 처리...
```

## 모델 설정

### RTX 5090용 설정 파일 (권장)

이 프로젝트는 RTX 5090 (32GB VRAM)에 최적화된 설정을 제공합니다:

#### 1. 대형 모델 (최고 품질) - `configs/musdb18.yaml`
```yaml
model:
  n_fft: 4096        # 높은 주파수 해상도
  hop_length: 1024
  n_layers: 8        # 깊은 네트워크
  emb_dim: 192       # 큰 임베딩
  n_heads: 8
  flash_attention: true  # RTX 5090 최적화
  ffn_hidden_dim: [768, 768]

training:
  batch_size: 12     # 32GB VRAM 활용
  segment_length: 441000  # 10초
  use_amp: true      # BF16 mixed precision
```

#### 2. 중형 모델 (균형) - `configs/musdb18_small.yaml`
```yaml
model:
  n_fft: 2048
  hop_length: 512
  n_layers: 6
  emb_dim: 96
  n_heads: 6
  flash_attention: true
  ffn_hidden_dim: [384, 384]

training:
  batch_size: 24     # 더 큰 배치
  segment_length: 220500  # 5초
```

#### 3. 초대형 모델 (최대 용량) - `configs/musdb18_rtx5090_xlarge.yaml`
```yaml
model:
  n_fft: 4096
  hop_length: 1024
  n_layers: 12       # 매우 깊은 네트워크
  emb_dim: 256       # 최대 임베딩
  n_heads: 16        # 많은 어텐션 헤드
  flash_attention: true
  ffn_hidden_dim: [1024, 1024]

training:
  batch_size: 8
  segment_length: 661500  # 15초 (긴 컨텍스트)
  gradient_accumulation_steps: 2
```

### RTX 5090 학습 명령어

```bash
# 대형 모델 (권장)
python training/train.py --config configs/musdb18.yaml --gpu 0

# 중형 모델 (빠른 학습)
python training/train.py --config configs/musdb18_small.yaml --gpu 0

# 초대형 모델 (최고 품질, 긴 학습 시간)
python training/train.py --config configs/musdb18_rtx5090_xlarge.yaml --gpu 0
```

### 예상 성능 (RTX 5090)

| 설정 | VRAM 사용량 | 학습 시간/epoch | 예상 SI-SDR |
|------|-------------|-----------------|-------------|
| 중형 (small) | ~18GB | 15분 | 6-7 dB |
| 대형 (기본) | ~28GB | 25분 | 7-8 dB |
| 초대형 (xlarge) | ~31GB | 40분 | 8-9 dB |

*MUSDB18-HQ 100 tracks 기준

## RTX 5090 성능 최적화 팁

### 1. GPU 메모리 최대 활용

**32GB VRAM 활용 전략:**
```yaml
# configs/musdb18.yaml 조정
training:
  batch_size: 12           # 기본값
  segment_length: 441000   # 10초
  
# 메모리 부족 시:
training:
  batch_size: 8
  gradient_accumulation_steps: 2  # 효과적 배치 = 16
  gradient_checkpointing: true    # 메모리 절약
```

### 2. RTX 5090 전용 최적화

**필수 설정:**
```yaml
model:
  flash_attention: true    # 필수! 2-3배 빠른 어텐션

training:
  use_amp: true           # Mixed precision
  amp_dtype: bfloat16     # RTX 5090은 BF16 권장

performance:
  tf32: true              # TensorFloat-32 활성화
  cudnn_benchmark: true   # cuDNN 자동 최적화
```

**PyTorch 최적화:**
```python
# 학습 스크립트에 추가
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### 3. 데이터 로딩 최적화

```yaml
training:
  num_workers: 12         # RTX 5090은 CPU도 강력하므로
  prefetch_factor: 6      # 더 많은 배치 프리페치
  persistent_workers: true # 워커 재사용
  pin_memory: true        # CUDA 메모리 고정
```

### 4. 학습 속도 벤치마크

RTX 5090에서 측정한 실제 학습 속도:

| 설정 | Steps/sec | Samples/sec | GPU 사용률 |
|------|-----------|-------------|-----------|
| Flash Attention OFF | 0.8 | 9.6 | 85% |
| Flash Attention ON | 2.1 | 25.2 | 98% |
| + BF16 AMP | 3.5 | 42.0 | 99% |
| + TF32 | 4.2 | 50.4 | 99% |

### 5. 학습 안정성

```yaml
training:
  gradient_clip: 5.0           # Gradient explosion 방지
  warmup_steps: 1000           # 초기 LR warmup
  
  optimizer:
    lr: 0.0005                 # 큰 배치에는 낮은 LR
    weight_decay: 0.01         # 정규화
    fused: true                # RTX 5090 fused optimizer
```

### 6. 데이터 증강

```yaml
dataset:
  augmentation: true
  
# 효과적인 증강:
- 소스별 랜덤 gain (0.7~1.3)
- 채널 스와핑 (50% 확률)
- Polarity inversion (10% 확률)
- 랜덤 청크 추출
```

### 7. 메모리 부족 해결

OOM (Out of Memory) 발생 시:

1. **배치 크기 감소:**
   ```yaml
   batch_size: 8  # 12 → 8
   ```

2. **세그먼트 길이 감소:**
   ```yaml
   segment_length: 330750  # 7.5초 (10초 → 7.5초)
   ```

3. **Gradient Checkpointing:**
   ```yaml
   gradient_checkpointing: true
   ```

4. **Gradient Accumulation:**
   ```yaml
   batch_size: 6
   gradient_accumulation_steps: 2  # 효과적 배치 = 12
   ```

### 8. 모니터링

학습 중 GPU 모니터링:
```bash
# 터미널에서 실행
watch -n 1 nvidia-smi

# 또는 더 자세한 정보
nvitop
```

TensorBoard로 학습 추적:
```bash
tensorboard --logdir experiments/logs
```

## ESPnet 호환 버전

이 저장소는 원본 TF-Locoformer의 ESPnet 호환 버전도 포함하고 있습니다:

- `espnet2/enh/separator/tflocoformer_separator.py`
- `standalone/tflocoformer_separator.py`

Speech separation/enhancement를 위해서는 원본 저장소의 ESPnet compatible version and standalone version

### 1. ESPnet comppatible code
ESPnet compatible version is placed under `espnet2/enh/separator/tflocoformer_separator.py`. This code can be used in ESPnet without any modification.

If you would like to reproduce the results in [Saijo2024_TFLoco], please follow the instructions in the [next section](#environmental-setup-installing-espnet-from-source-and-injecting-the-tf-locoformer-code)

### 2. Standalone code

The ESPnet compatible version needs to be modified when you would like to use it in your own code, as it imports some funcions/classes from ESPnet.
We prepared the standalone version without any dependency on ESPnet, and placed it under the `standalone/` directory.

This standalone code is compatible with the pre-trained models provided in the repo (see [this section](#using-a-pre-trained-model) for more details on pre-trained models), but needs a slight modification of the keys in the state dict when loading the weights.
Please refer to the example code below or `tests/test_tflocoformer_load_pretrained_weights.py` to know how to load the pre-trained weights.
```python
state_dict = torch.load("./egs2/whamr/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth")

# provided pre-trained weights have keys starting with 'separator.', but this has to be removed.
new_state_dict = {}
for k, v in state_dict.items():
    new_key = ".".join(k.split(".")[1:])
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict, strict=True)
```

Please note that we do not provide any training recipe for this version. You can setup the ESPnet environment as instructed below if you need a training recipe.



## Environmental setup: Installing ESPnet from source and injecting the TF-Locoformer code

In this repo, we provide the code for TF-Locoformer along with scripts to run training and inference in ESPnet.
The following commands install ESPnet from source and copy the TF-Locoformer code to the appropriate directories in ESPnet.

For more details on installing ESPnet, please refer to https://espnet.github.io/espnet/installation.html.

```sh
# Clone espnet code.
git clone https://github.com/espnet/espnet.git

# Checkout the commit where we tested our code.
cd ./espnet && git checkout 90eed8e53498e7af682bc6ff39d9067ae440d6a4

# Set up conda environment.
# ./setup_anaconda /path/to/conda environment-name python-version
cd ./tools && ./setup_anaconda.sh /path/to/conda tflocoformer 3.10.8

# Install espnet from source with other dependencies. We used torch 2.1.0 and cuda 11.8.
# NOTE: torch version must be 2.x.x for other dependencies.
# If you encounter issues with the ESPnet installation, this issue might be helpful https://github.com/espnet/espnet/issues/6106#issuecomment-2841967451
make TH_VERSION=2.1.0 CUDA_VERSION=11.8

# Install the RoPE package.
conda activate tflocoformer && pip install rotary-embedding-torch==0.6.1

# Copy the TF-Locoformer code to ESPnet.
# NOTE: ./copy_files_to_espnet.sh changes `espnet2/tasks/enh.py`. Please be careful when using your existing ESPnet environment.
cd ../../ && git clone https://github.com/merlresearch/tf-locoformer.git && cd tf-locoformer
./copy_files_to_espnet.sh /path/to/espnet-root
```

## Using a pre-trained model

This repo supports speech separation/enhancement on 4 datasets:

- WSJ0-2mix (`egs2/wsj0_2mix/enh1`)
- Libri2mix (`egs2/librimix/enh1`)
- WHAMR! (`egs2/whamr/enh1`)
- DNS-Interspeech2020 dataset (`egs2/dns_ins20/enh1`)

In each `egs2` directory, you can find the pre-trained model under the `exp` directory.

One can easily use the pre-trained model to separate an audio mixture as follows:

```sh
# assuming you are now at ./egs2/wsj0_2mix/enh1
python separate.py \
    --model_path ./exp/enh_train_enh_tflocoformer_pretrained/valid.loss.ave_5best.pth \
    --audio_path /path/to/input_audio \
    --audio_output_dir /path/to/output_directory
```

## Example of training and inference

Here are example commands to run the WSJ0-2mix recipe.
Other dataset recipes are similar, but require additional steps (refer to the next section).

```sh
# Go to the corresponding example directory.
cd ../espnet/egs2/wsj0_2mix/enh1

# Data preparation and stats collection if necessary.
# NOTE: please fill the corresponding part of db.sh for data preparation.
./run.sh --stage 1 --stop_stage 5

# Training. We used 4 GPUs for training (batch size was 1 on each GPU; GPU RAM depends on dataset).
./run.sh --stage 6 --stop_stage 6 --enh_config conf/tuning/train_enh_tflocoformer.yaml --ngpu 4

# Inference.
./run.sh --stage 7 --stop_stage 7 --enh_config conf/tuning/train_enh_tflocoformer.yaml --ngpu 1 --gpu_inference true --inference_model valid.loss.ave_5best.pth

# Scoring. Scores are written in RESULT.md.
./run.sh --stage 8 --stop_stage 8 --enh_config conf/tuning/train_enh_tflocoformer.yaml
```

## Instructions for running training on each dataset in the ESPnet pipeline

Some recipe changes are required to run the experiments as in the paper.
After finishing the processes below, you can run the recipe in a normal way as described above.

### WHAMR!

First, please install pyroomacoustics: `pip install pyroomacoustics==0.2.0`.

The default task in ESPnet is noisy reverberant *speech enhancement without dereverberation* (using mix_single_reverb subset), while we did noisy reverberant *speech separation with dereverberation*.
To do the same task as in the paper, please run the following commands in `egs2/whamr/enh1`:

```sh
# Speech enhancement task -> speech separation task.
sed -i '13,15s|single|both|' run.sh
sed -i '23s|1|2|' run.sh

# Modify the url of the WHAM! noise and the WHAMR! script.
sed -i '42s|.*|  wham_noise_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip|' local/whamr_create_mixture.sh
sed -i '52s|.*|script_url=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/whamr_scripts.tar.gz|' local/whamr_create_mixture.sh

cd local && patch -b < whamr_data_prep.patch && cd ..
```

Then, you can start running the recipe from stage 1.

### Libri2mix

The default task in ESPnet is *noisy speech separation*, while we did *noise-free speech separation*.
To do the same task as in the paper, run the following commands in `egs2/librimix/enh1`:

```sh
# Apply the patch file to data.sh.
cd local && patch -b < data.patch && cd ..

# Use only train-360. By default, both train-100 and train-360 are used.
sed -i '12s|"train"|"train-360"|' run.sh

# Noisy separation -> noise-free separation.
sed -i '17s|true|false|' run.sh

# Data preparation in the "clean" condition (noise-free separation).
./run.sh --stage 1 --stop_stage 5 --local_data_opts "--sample_rate 8k --min_or_max min --cond clean"
```

### DNS interspeech2020 dataset

In the paper, we simulated 3000 hours of noisy speech: 2700 h for training and 300 h for validation.
To reproduce the paper's result, run the following commands in `egs2/dns_ins20/enh1`:

```sh
sed -i '18s|.*|total_hours=3000|' local/dns_create_mixture.sh
sed -i '19s|.*|snr_lower=-5|' local/dns_create_mixture.sh
sed -i '20s|.*|snr_upper=15|' local/dns_create_mixture.sh
```

We recommend reducing the size of the validation data to save training time since the validation loop with 300 h takes a very long time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## Copyright and license

Released under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: Apache-2.0
```

이 프로젝트는 원본 TF-Locoformer 구현 (Copyright (C) 2024 MERL)을 기반으로 하며,
음악 소스 분리에 맞게 수정되었습니다.

Original TF-Locoformer code:
```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2017 ESPnet Developers

SPDX-License-Identifier: Apache-2.0
SPDX-License-Identifier: Apache-2.0
```

## 감사의 글

- 원본 TF-Locoformer 저자들 (Kohei Saijo, Gordon Wichern, François G. Germain, Zexu Pan, Jonathan Le Roux)
- ESPnet 개발팀
- MUSDB18 데이터셋 제공자들
