# EIP 연구 — 작업 규약

SNN 스파이크 규제 연구 (CVPR 투고 목표).
**목표: 정확도를 희생하지 않으면서 스파이크를 줄인다.**

## 승인 등급 — 이게 가장 중요하다

에이전트가 무엇을 하느냐보다 **어디까지 사용자 승인 없이 하느냐**가 먼저다.

**1등급 — 자유 (읽기만)**
결과 집계, 통계 계산, 로그·코드 읽기, 레퍼런스 검색, 반론 검토.

**2등급 — 파일은 쓰되 반영은 안 함**
코드 수정, 그림 생성, 논문 초안. 파일을 쓰고 **멈춘다.** 커밋하지 않는다.

**3등급 — 반드시 사용자 승인. 제안하고 멈춘다**
- `git push` / 커밋 (어떤 브랜치든)
- 돌고 있는 실험 kill, 실행기 기동·중지
- 큐(`_queue/QUEUE.tsv`) 상태 변경 — 재투입·취소 포함
- 파일·디렉토리 삭제
- 원격 서버(sejong) 상태 변경 — pull, env 변경, 실행기 기동
- 학습 경로 4파일 변경의 **배포**

**자율 루프 스킬(`ralph`, `autopilot`, `ultrawork`, `autoresearch`)은 쓰지 않는다.**
사람 확인 없이 계속 도는 물건이고, 지금은 push 가 곧 배포다.

## 왜 이렇게까지 하나

| 날짜 | 원인 | 피해 |
|---|---|---|
| 2026-09-14 | 학습 코드 한 줄 고치고 연기시험 생략 | **17런 사망** |
| 2026-09-16 | 원격 서버가 `git pull` 안 함 | **7런 사망** |
| 2026-09-15 | `queue_runner` kill → 자식이 고아 | 5런 기록 유실 |

한 런이 30시간이다. 빠른 것보다 안 죽는 게 중요하다.

## 에이전트 분담과 쓰기 권한

| 에이전트 | 모델 | 쓰기 가능 |
|---|---|---|
| (메인 세션) | Opus | 없음 — 계획·승인·보고 |
| `eip-ops` | Sonnet | `_queue/`, 실행기 (3등급은 승인) |
| `eip-code` | Sonnet (학습 경로는 `model=opus`) | 코드 (커밋 금지) |
| `eip-stats` | Opus | `analysis/`, `docs/paper/`, 스펙 |
| `eip-fig` | Sonnet | `EIP_fig/` |
| `eip-paper` | Opus | `paper/`, `docs/paper/` |
| `eip-refs` | Sonnet | `docs/paper/refs-audit.md` |
| `eip-critic` | Opus | 없음 (읽기 전용) |

**작성자와 검증자를 같은 컨텍스트에 두지 않는다.** 논문 초안은 `eip-critic` 을 거친다.

## 단일 진실 원천

수치는 **`eip-stats` 만** 만든다. 그림도 논문도 그 산출물과 스펙
(`docs/superpowers/specs/2026-09-14-eip-paper-spec.md`)만 읽는다.
원본 실험 CSV 를 각자 읽으면 n 이 어긋난다 — 실제로 `ours 3e-3` 이 n=4 와 n=7 로 갈렸다.

## Codex 의 위치

격리된 새 작업과 2차 코드 리뷰에만 쓴다.
- 맡길 것: 새 분석 스크립트, `s_count`/npz 불일치 디버깅, PyTorch 재현(별도 env·폴더),
  학습 경로 변경의 2차 리뷰
- **맡기지 않을 것: 학습 경로 4파일, 큐·실행기.** Codex 는 "지금 16런이 돌고 있다"는
  맥락을 갖지 않는다

## 학습 경로 — 여기를 고치면 배포다

```
lib_snn/neurons.py   flags.py   run_paper.py   config_snn_training.py
```
두 서버가 같은 브랜치를 pull 한다. 변경 후에는 **2에폭 연기시험**을 통과해야 한다.
`rc=0` 만으로 부족하다 — 체크포인트(`*.weights.h5`)가 실제로 생겼는지 센다.

## 환경

- **canus** = 161.122.23.138 (표기 `138`). 산출물 `/media/hdd1/kyccj/EIP/paper`,
  python `~/anaconda3/envs/venv_1/bin/python`
- **sejong** = 161.122.23.23 (표기 `23`), ssh 포트 **23456**.
  산출물 `/srv2/kyccj/EIP/paper`, python `~/anaconda3/envs/venv_eip/bin/python`.
  `EIP_STORE`/`EIP_PYTHON` export 필요
- **canus GPU 6,7 은 juyun 것이다. 절대 쓰지 않는다.**
- 실행기를 늘릴 때 기존 것을 kill 하지 않는다. **두 번째 실행기를 붙인다** (kill 하면 고아)
- `pkill -f <패턴>` 금지 — 자기 셸이 죽는다. **`kill` 에는 정확한 PID 만 넘긴다.**
  `for p in $(ps | grep 패턴)` 처럼 grep 결과를 kill 에 흘리는 것도 같은 사고다 (09-19, exit 144).
  절차: (1) 읽기 전용 명령으로 PID 를 눈으로 확인 → (2) 다음 명령에서 `kill <숫자>`.

## 보고 규약

- 표 열 순서 고정: `train loss → train acc → val loss → val acc → spikes`
- 반복 런은 **개별값 전부 + 그룹 바로 아래 평균 행**. 평균만 보여주지 않는다
- 서버 열을 넣는다 (`138` / `23`)
- 방법 이름은 영어: `baseline`, `ours`, `plain L2`, `BPSR`, `1-softmax`.
  ablation 은 `- vmem`, `- final_step`, `- 1- inversion`, `- loss-ratio`
- **경쟁 범위 세 팔의 이름 (2026-09-21 확정).** `reg_spike_maxnorm_group` 의 내부값을
  글·표·그림에 쓰지 않는다. 채널 **안**에서 겨루면 intra, 채널 **끼리** 겨루면 inter 다.

  | 이름 | 코드 플래그 | 겨루는 상대 | 옛 표기 |
  |---|---|---|---|
  | `ours_intra_ch` | `within_channel` | 같은 채널 안의 다른 위치 | `ours` / `prop` |
  | `ours_layer` | `none` | 층의 모든 뉴런 | `ours_layer` |
  | `ours_inter_ch` | `channel` | 다른 채널 (채널 공간합끼리) | `ours_ch` |

  **원래 방법(제안법)은 `ours_intra_ch` 다.** 런 디렉토리명·`run_paper.py` 의 방법 키는
  기존대로 `prop`/`ours_ch` 를 쓴다 (바꾸면 완주한 런과 큐가 어긋난다).
- 그래프는 **실제 정확도(y) × 실제 스파이크 수(x)**. 차이(delta)로 그리지 않는다
- `1-maxnorm` = `1 − sc/max`, `maxnorm` = `sc/max`
- **모든 수치에 n 을 붙인다.** 정확도 시드 간 SD 는 0.10~0.13%p — 그보다 작은 차이는
  n=2 로 보이지 않는다
- 간결하게. 조어를 만들지 않고, 새 용어는 정의를 병기한다

## 커밋

`Co-Authored-By: Claude` / `Claude-Session:` 트레일러를 **넣지 않는다.**
실험 산출물 디렉토리와 `.omc/` 는 커밋하지 않는다.

## 결정 기록

확정·기각된 것은 `docs/decisions.md` 에 누적한다. 작업 시작 전에 읽는다.
