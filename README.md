# Detecting Negative Transfer in Multi-Agent Coordination Games

온라인 MARL에서 source task에서 배운 협동 skill이 target task 학습에 도움이 되는지, 혹은 오히려 해로운 negative transfer를 일으키는지 탐지하는 프로젝트입니다.

핵심 질문:

> MARL에서 transfer할 skill과 버릴 skill을 target 초반 경험만으로 구분할 수 있는가?

## 폴더 구조

```text
Paper_3/
├── README.md
├── configs/
│   ├── env/
│   │   ├── lbf/
│   │   ├── overcooked/
│   │   └── mpe/
│   ├── algo/
│   ├── transfer/
│   └── experiment/
├── src/
│   └── negtransfer_marl/
│       ├── envs/
│       ├── algorithms/
│       │   ├── ippo/
│       │   ├── mappo/
│       │   └── qmix/
│       ├── skills/
│       ├── transfer/
│       ├── runners/
│       ├── analysis/
│       └── utils/
├── scripts/
├── checkpoints/
│   ├── source_policies/
│   └── skill_bank/
├── runs/
│   ├── lbf/
│   ├── overcooked/
│   └── mpe/
├── results/
│   ├── tables/
│   ├── figures/
│   └── summaries/
├── tests/
└── paper/
```

## 디렉터리 설명

### `configs/`

실험 설정 파일을 모아두는 폴더입니다.

- `configs/env/`: 환경별 설정
  - `lbf/`: Level-Based Foraging 실험 설정
  - `overcooked/`: Overcooked-AI 실험 설정
  - `mpe/`: PettingZoo MPE/SISL 계열 sanity check 설정
- `configs/algo/`: IPPO, MAPPO, QMIX 등 알고리즘 설정
- `configs/transfer/`: scratch, full transfer, random gate, negative transfer gate 등 transfer 방식 설정
- `configs/experiment/`: 여러 환경/알고리즘/transfer 설정을 묶은 전체 실험 설정

### `src/negtransfer_marl/`

실제 Python 패키지 소스 코드입니다.

- `envs/`: LBF, Overcooked, PettingZoo wrapper와 task specification
- `algorithms/`: MARL 학습 알고리즘 구현
  - `ippo/`: Independent PPO
  - `mappo/`: Multi-Agent PPO
  - `qmix/`: QMIX 계열 value decomposition
- `skills/`: source policy를 skill/option 형태로 저장하고 불러오는 코드
- `transfer/`: negative transfer detector, gating network, skill probing, transfer metric 코드
- `runners/`: source 학습, target 학습, 평가, sweep 실행 entrypoint
- `analysis/`: learning curve, skill usage, transfer matrix, 통계 분석 코드
- `utils/`: seed 고정, logging, checkpoint 저장/로드 등 공통 유틸

### `scripts/`

반복 실행할 실험 명령어를 shell script로 저장하는 폴더입니다.

예:

- source task policy 학습
- target task transfer 실험
- ablation 실험
- figure/table 생성

### `checkpoints/`

학습된 모델과 skill bank를 저장합니다.

- `source_policies/`: source task에서 학습한 policy checkpoint
- `skill_bank/`: source policy에서 추출하거나 저장한 latent option / skill checkpoint

### `runs/`

학습 중 생성되는 raw log를 저장합니다.

- `runs/lbf/`: LBF 실험 로그
- `runs/overcooked/`: Overcooked 실험 로그
- `runs/mpe/`: MPE/SISL 실험 로그

### `results/`

raw log를 가공한 최종 결과물을 저장합니다.

- `tables/`: 논문용 표
- `figures/`: learning curve, skill usage plot 등 그림
- `summaries/`: 실험별 요약 결과

### `tests/`

환경 wrapper, detector, gating, 재현성 관련 테스트 코드를 둡니다.

### `paper/`

논문 작성용 메모와 초안을 저장합니다.

예:

- related work 정리
- method 설명 초안
- experiment design 메모
- 결과 해석 메모

## 환경 설정

이 프로젝트는 `uv`로 Python 환경을 관리합니다. 기본 설정은 JAX CUDA 12 pip wheel을 사용해 NVIDIA GPU에서 실행되도록 구성되어 있습니다.

```bash
uv sync --extra dev
uv run python scripts/check_env.py
uv run pytest -q
```

GPU 인식 여부는 다음처럼 확인할 수 있습니다.

```bash
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

현재 검증 환경에서는 `gpu` backend와 `CudaDevice(id=0)`가 확인되었습니다. `jaxmarl==0.1.0`이 `jax<=0.4.38`을 요구하므로, CUDA 13 wheel 대신 `jax[cuda12]==0.4.38`을 사용합니다.

## Overcooked 환경 구현 상태

`docs/overcooked_environment_pairs_ko.md`의 Pair A–E를 JaxMARL classic Overcooked 환경으로 생성할 수 있도록 구현했습니다.

- layout/pair registry: `src/negtransfer_marl/envs/overcooked_layouts.py`
- pair config: `configs/env/overcooked/pairs.yaml`
- default env config: `configs/env/overcooked/default.yaml`
- pair smoke check: `scripts/check_overcooked_pairs.py`
- regression tests: `tests/test_overcooked_layouts.py`

전체 pair 생성 검증:

```bash
uv run python scripts/check_overcooked_pairs.py
uv run pytest -q
```

## 현재 구현 상태와 다음 단계

현재 완료된 것은 **환경 세팅 및 Overcooked 환경 구현**입니다.

완료된 항목:

- `uv` 기반 Python 환경 세팅
- JAX GPU 실행 설정
  - 검증 backend: `gpu`
  - 검증 device: `CudaDevice(id=0)`
- JaxMARL classic Overcooked 실행 확인
- `docs/overcooked_environment_pairs_ko.md`의 Pair A–E 구현
  - Pair A: `forced_coordination -> forced_coordination_reversed`
  - Pair B: `forced_coordination -> coordination_ring`
  - Pair C: `forced_coordination -> open_parallel_room`
  - Pair D: `bottleneck -> wide_bottleneck`
  - Pair E: `coordination_ring -> counter_circuit_o_1order`
- Overcooked-AI style grid를 JaxMARL layout으로 변환하는 parser 구현
- agent `1`/`2` identity 보존
- source-target pair registry 구현
- 환경 config, smoke script, regression test 추가

관련 파일:

```text
src/negtransfer_marl/envs/overcooked_layouts.py
configs/env/overcooked/default.yaml
configs/env/overcooked/pairs.yaml
scripts/check_env.py
scripts/check_overcooked_pairs.py
tests/test_environment_setup.py
tests/test_overcooked_layouts.py
```

검증 명령:

```bash
uv sync --extra dev --frozen
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
uv run python scripts/check_env.py
uv run python scripts/check_overcooked_pairs.py
uv run pytest -q
uv run ruff check pyproject.toml scripts tests src
```

현재 검증 결과:

```text
gpu
[CudaDevice(id=0)]
16 passed
All checks passed!
```

### 아직 구현되지 않은 부분

`src/negtransfer_marl/baselines/`에 baseline 파일만 추가한다고 바로 알고리즘을 실행할 수 있는 상태는 아닙니다. 현재는 환경 생성과 검증까지만 완료되어 있고, 실제 학습/transfer 실행을 위한 알고리즘 및 runner는 아직 필요합니다.

아직 필요한 항목:

- IPPO/MAPPO/QMIX 학습 코드
- rollout collector
- PPO/MAPPO update loop
- training runner
- config loader
- checkpoint 저장/로드
- source policy training
- baseline별 실행 로직
- learning curve 저장
- behavior diagnostics 계산 코드
- CAST selective transfer model/gate 구현

최소 실행 가능 구조는 다음과 같이 잡을 수 있습니다.

```text
src/negtransfer_marl/
├── baselines/
│   ├── scratch.py
│   ├── full_transfer.py
│   ├── encoder_only.py
│   └── skill_only.py
├── algorithms/
│   └── ippo/
│       ├── networks.py
│       ├── rollout.py
│       └── train.py
├── runners/
│   └── train_overcooked.py
└── envs/
    └── overcooked_layouts.py
```

목표 실행 예시는 다음과 같습니다.

```bash
uv run python -m negtransfer_marl.runners.train_overcooked \
  --pair pair_a \
  --method scratch \
  --seed 0
```

### 추천 구현 순서

먼저 **Scratch IPPO baseline**부터 구현하는 것을 권장합니다. 이 단계가 완료되어야 다른 transfer baseline과 CAST를 안정적으로 비교할 수 있습니다.

권장 순서:

1. `algorithms/ippo/`에 기본 IPPO/PPO 네트워크와 학습 루프 구현
2. `baselines/scratch.py`에서 source transfer 없이 target 학습 정의
3. `runners/train_overcooked.py`에서 pair/env/method/seed를 받아 실행
4. `runs/overcooked/...`에 learning curve와 config 저장
5. `checkpoints/source_policies/...`에 source policy 저장
6. `baselines/full_transfer.py` 구현
7. `baselines/encoder_only.py` 구현
8. `baselines/skill_only.py` 구현
9. behavior diagnostics 구현
10. CAST selective transfer 구현

즉, 현재 상태는 **환경은 준비 완료**, 다음 단계는 **학습 runner와 Scratch IPPO baseline 구현**입니다.
