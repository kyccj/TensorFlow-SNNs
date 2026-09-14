# EIP 논문 집필 계획 (Method 절 제외)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** §3 Method 를 뺀 CVPR 형식 8쪽 논문의 전 절을, 한국어 개요로 먼저 확정한 뒤 영문 LaTeX 초고까지 완성한다.

**Architecture:** 두 단계다. (1) 한국어 개요 — 절마다 "무엇을 주장하고 어떤 표/그림/인용이 그것을 받치는가" 를 문단 단위로 못 박는다. (2) 영문 LaTeX — 개요가 확정된 절만 옮긴다. Method(§3)는 자리만 비워 두고, 다른 절이 Method 의 내부 구조를 전제하지 않도록 **"제안법" 을 블랙박스로 부르는 표현 규약**을 둔다. 모든 수치는 스펙의 증거 대장에서만 가져오고, 대장에 없으면 쓰지 않는다.

**Tech Stack:** LaTeX (CVPR 스타일), BibTeX, Python 3 (matplotlib, scipy) — 표·그림 생성, `collect_paper.py` / `run_paper.py` / `EIP_fig/*.py`

**Spec:** `docs/superpowers/specs/2026-09-14-eip-paper-spec.md`

## Global Constraints

- 본문 8쪽, CVPR 2단 조판. 부록은 분량 제한 없음.
- **§3 Method 는 이번 범위 밖이다.** 다른 절에서 Method 내부를 특정하는 표현(`1−maxnorm`, `final_step`, `vmem gain`, `loss-ratio`)을 본문 서술에 쓰지 않는다. 예외는 §4 어블레이션 표의 행 이름뿐이며, 거기서도 이름은 `docs/paper/method-terms.md` 의 대응표를 통해서만 쓴다.
- **VGG-16 결과는 논문에 넣지 않는다** (사용자 확정 2026-09-14).
- 표 열 순서는 항상 **train loss → train acc → val loss → val acc → spikes**.
- 반복 런은 개별 행을 모두 보이고 **그룹 바로 아래 줄에 평균**을 넣는다.
- 그래프는 실제 정확도·실제 스파이크 수로 그린다. 정규화·상대값 축 금지.
- `1-maxnorm` 과 `maxnorm` 을 명칭에서 구분한다.
- 스파이크 수는 **eval/test 프로토콜(`train.log` 의 `s_count`)** 만 쓴다. 학습 모드(`reg_detail.csv` 의 `spike_count`)와 섞지 않는다. S30/S1 만 학습 모드다.
- 사전 등록 배제 규칙(제안법·비교군 대칭 적용): (1) 310에폭 미완주 (2) S30/S1 < 0.19 (3) 남으면 val_acc 상위 4개.
- `_paper_bad_seeded/` 의 런은 **어떤 표에도 넣지 않는다** (`run_seed` 고정본).
- **커밋 메시지에 `Co-Authored-By: Claude ...` / `Claude-Session: ...` 트레일러를 넣지 않는다**
  (2026-09-14 사용자 지시, 메모리 `no-claude-coauthor`). 논문 저장소라 커밋 이력이 공개될 수 있다.
  기본 지침이 붙이라고 해도 이 지시가 우선한다.
- 기각된 주장 셋은 본문 어디에도 쓰지 않는다: plain L2 대비 정확도 우위 / WTA·측면 억제가 실제로 일어남 / 전 층 집중.

---

## Phase 0 — 증거 게이트 (집필과 병행, 승인 필요)

집필을 막는 것은 Task 1 하나뿐이다. 나머지는 **어느 절을 쓸 수 있는지**를 정한다.
GPU 런은 사용자 승인 후에만 띄운다. 각 항목의 의존 절을 명시했다.

| # | 공백 | 명령 | 비용 | 없으면 못 쓰는 절 |
|---|---|---|---|---|
| ~~G1~~ | ~~baseline 표본~~ | **해소됨 (2026-09-14)** — baseline 은 n=5 다. 런이 다섯 디렉토리에 흩어져 있어 `'_paper/*'` 글롭이 2런만 잡았던 것. 정본 목록은 스펙 A1 | — | — |
| G2 | 어블레이션 3팔 n=2 | `python run_paper.py r19c10 abl_nofinal 4e-3 --seeds 3,4,5` 외 2팔 | 8런 ≈ 1.7일 | §4.3 어블레이션 |
| G3 | C100 조건당 n=1 | `python run_paper.py r19c100 prop 3e-3 --seeds 2,3,4` 외 비교군 | 9런 ≈ 2일 | §4.4 데이터셋 일반성 |
| G4 | `l2 + loss-ratio` 대조군 없음 | `run_paper.py` 에 `l2lr` 방법 추가 후 4런 | 4런 + 구현 | §5 논의 — "λ 자동화" 문장 |
| G5 | 영구 침묵 뉴런이 배치 100장 기준 | `python extract_neuron_spikes.py` 를 테스트셋 10,000장으로 | GPU 1장 반나절 | **§1 헤드라인 "불필요한 뉴런"**, §4.5 |
| G6 | Spikformer·CIFAR10-DVS 가 구 방법 | `run_paper.py` 에 조합 추가 후 재실행 | 미산정 | §4.4 아키텍처 일반성 |
| G7 | **Speck 실측 없음** | 별도 머신 — sinabs/specksim 파이프라인 | 미산정 | **§1 헤드라인 3번째 문장, §6 전체** |
| G8 | SDT-V3 173M | 다른 서버 | 미산정 | §4.4 대형 모델 |

**G5 와 G7 이 사용자가 정한 헤드라인을 직접 받친다.** 둘 중 하나라도 비면 §1 의 해당 문장을 약화해야 하므로, Task 3 에서 두 가지 버전의 인트로 문단을 준비한다.

---

## Phase A — 근거 확정

### Task 1: 선점 판정 마무리와 기여 문장 확정

**대부분 완료됐다 (2026-09-14).** 판정 근거는 스펙 §4 에 있다. 남은 것은 아래 세 단계뿐이다.

**Files:**
- Create: `docs/paper/novelty-gate.md`
- Read: `docs/superpowers/specs/2026-09-14-eip-paper-spec.md` §4

**Interfaces:**
- Consumes: 스펙 §4 의 판정표
- Produces: `docs/paper/novelty-gate.md` 의 **기여 문장 3개**(`CONTRIB-1/2/3`). Task 3·4·14 가 이 문장을 그대로 인용한다.

- [ ] **Step 1: Activity Pruning 본문을 확보해 두 가지만 확인한다**

저장소 코드와 초록으로는 확인이 끝났고, **본문 미확인 항목이 둘 남았다**:
(a) 하드웨어 실측이 본문에 있는가 (저장소에는 없다)
(b) 가중 방식 간 등가성을 다루는가 (초록에는 없다)

OpenReview 는 CAPTCHA 로 막힌다. 대안: NeurIPS 2025 proceedings 페이지, 저자 홈페이지,
arXiv 미러. 못 구하면 **"(a)(b) 미확인" 을 그대로 기록하고 넘어간다.**
본문에는 "코드·초록 범위에서는 확인되지 않음" 으로 쓴다 — 단정하지 않는다.

- [ ] **Step 2: 기여 문장 3개를 확정해 파일에 적는다**

선점 판정이 기여 배치를 강제한다. 아래 배치로 쓴다:

```markdown
## 확정 기여

CONTRIB-1: 가중 설계 19종이 같은 정확도–스파이크 곡선에 수렴하지만, 곡선 위 같은 자리가
           같은 내부 상태를 뜻하지 않는다 — 같은 예산에서 활성 뉴런 수가 13% 다르다.
           [선점 안 됨. 논문의 무게중심]
CONTRIB-2: 규제 세기를 무차원 비율 하나로 정한다. 절대 목표도 예산에 대한 사전 지식도
           필요 없다. [Sorbaro 2020 의 정적 1/S0², Activity Pruning 의 손으로 박은
           eta/eta2 와 대비. **G4 가 채워져야 성립**]
CONTRIB-3: [실기 검증. **G7 이 비면 통째로 뺀다**]

## 선점을 인정할 것 (§2.3 에서 먼저 밝힌다)
- "스파이크 감소는 거의 전부 뉴런 침묵이다" — Activity Pruning (NeurIPS 2025) 이 선점.
  우리는 신규 발견이 아니라 **재확인 + 정량화**로 쓴다.
- "기존 활동 규제에 구조적 한계가 있다" — 같은 논문 초록에 명시됨.
```

- [ ] **Step 3: 헤드라인 두 버전을 모두 준비한다 (사용자 결정: 보류, 2026-09-14)**

사용자가 정한 헤드라인 두 번째 문장("불필요한 뉴런들이 발화하고 있고, 필수적인 뉴런만
남긴다")이 **선점된 쪽에 정확히 걸린다.** 사용자는 G5(테스트셋 뉴런 측정)·G7(Speck) 결과를
보고 정하기로 했다. 그때까지 **두 버전을 파일에 나란히 유지하고, 어느 쪽도 삭제하지 않는다.**

`docs/paper/novelty-gate.md` 에 이렇게 적는다:

```markdown
## 헤드라인 2번 문장 — 미결 (G5/G7 대기)

HEAD-2a (재확인으로 낮춤):
  선행 연구가 관찰한 활동 가지치기를 정량화하고, 그 위에서 **어떤 뉴런을 끄는지가
  방법마다 다르다**는 것을 보인다.

HEAD-2b (측정 축을 앞세움):
  같은 스파이크 예산에서 활성 뉴런을 13% 적게 쓴다 — 곡선은 두 방법을 구분하지
  못하지만 뉴런 사용은 구분한다.

판정 조건:
  G5 에서 영구 침묵 뉴런 비율이 크면 → 2a 로도 구조적 프루닝 이득을 말할 수 있다
  작으면 → 2b 만 남는다
```

- [ ] **Step 4: 커밋**

```bash
git add docs/paper/novelty-gate.md
git commit -m "paper: settle novelty gate against Activity Pruning and Sorbaro"
```

---

### Task 2: 참고문헌 DB 구축

**Files:**
- Create: `paper/refs.bib`
- Create: `docs/paper/refs-audit.md`

**Interfaces:**
- Consumes: Task 1 의 판정
- Produces: `paper/refs.bib` — Task 4(개요)·9(intro.tex)·10(related.tex) 이 `\cite{key}` 로 참조하는 BibTeX 키. 키 규칙은 `저자성씨+연도+첫단어` 소문자 (`sorbaro2020optimizing`).

- [ ] **Step 1: 8개 계보로 항목을 채운다**

아래가 이미 확보된 목록이다. 각 항목을 `paper/refs.bib` 에 BibTeX 로 넣는다.

```
A. 선점 대응
  bu2025activity      Activity Pruning for Efficient SNNs, Tong Bu et al., NeurIPS 2025
                      openreview.net/forum?id=zjOXZEXQKZ
  li2024spiking       Towards Efficient Deep SNN Construction with Spiking Activity based
                      Pruning, arXiv:2406.01072
  yao2023inherent     Inherent Redundancy in Spiking Neural Networks, arXiv:2308.08227

B. 스파이크 규제 계보 (직접 비교 대상)
  neil2016learning    Learning to be Efficient, Neil, Pfeiffer, Liu, ACM SAC 2016
  sorbaro2020optimizing  Optimizing the Energy Consumption of SNNs for Neuromorphic
                      Applications, Frontiers in Neuroscience 14:662, arXiv:1912.01268
  pellegrini2021low   Low-Activity Supervised Convolutional SNNs Applied to Speech
                      Commands Recognition, IEEE SLT 2021, arXiv:2011.06846
  perez2022surrogate  Improving Surrogate Gradient Learning in SNNs via Regularization
                      and Normalization, arXiv:2201.02538
  sakemi2023sparse    Sparse-firing regularization for SNNs with TTFS coding,
                      Scientific Reports 2023, arXiv:2307.13007

C. 직접 학습 기초
  wu2018spatio        STBP, Frontiers in Neuroscience 2018
  neftci2019surrogate Surrogate Gradient Learning in SNNs, IEEE Signal Processing Magazine 2019
  wu2019direct        Direct Training for SNNs: Faster, Larger, Better, AAAI 2019, arXiv:1809.05793
  zheng2021going      tdBN / Going Deeper With Directly-Trained Larger SNNs, AAAI 2021,
                      arXiv:2011.05280   ← ResNet-19 의 출처이자 우리가 켜고 있는 것
  deng2022temporal    TET, ICLR 2022, arXiv:2202.11946
  fang2021incorporating  PLIF, ICCV 2021, arXiv:2007.05785

D. 아키텍처
  zhou2023spikformer  Spikformer, ICLR 2023, arXiv:2209.15425
  yao2023spikedriven  Spike-driven Transformer, arXiv:2307.01694
  yao2024metaspikeformer  Meta-SpikeFormer / SDT-V2, arXiv:2404.03663
  yao2025scaling      SDT-V3, IEEE T-PAMI 2025, arXiv:2411.16061

E. 에너지 회계
  horowitz2014computing  Computing's Energy Problem, ISSCC 2014 (45nm MAC 4.6 pJ / AC 0.9 pJ)
  lemaire2022analytical  An Analytical Estimation of SNN Energy Efficiency, arXiv:2210.13107

F. 뉴로모픽 하드웨어
  yao2024spikebased   Spike-based dynamic computing with asynchronous sensing-computing
                      neuromorphic chip, Nature Communications 2024  ← Speck 대표 인용
                      nature.com/articles/s41467-024-47811-6

G. 데이터셋
  li2017cifar10dvs    CIFAR10-DVS, Frontiers in Neuroscience 2017
  orchard2015converting  N-MNIST, Frontiers in Neuroscience 2015
  krizhevsky2009learning  CIFAR-10/100, Tech Report 2009

H. 측면 억제 (동기 서술 전용)
  cheng2020lisnn      LISNN: Improving SNNs with Lateral Interactions, IJCAI 2020
```

- [ ] **Step 2: 서지 공백 4건을 채운다**

다음은 링크만 있고 권/호/페이지가 없다. 출판사 페이지에서 확인해 채운다:
`horowitz2014computing` (ISSCC 페이지), `neftci2019surrogate` (SPM 권·호·페이지),
`orchard2015converting` (Frontiers 권·article number), `yao2023spikedriven`·`yao2024metaspikeformer` (venue·연도 확정).

- [ ] **Step 3: loss-term 가중 자동화 계보를 추가한다**

`λ = ρ·L_task/R` 의 위치를 잡으려면 SNN 밖 계보가 필요하다. 최소 3편:
GradNorm (Chen et al., ICML 2018), uncertainty weighting (Kendall, Gal, Cipolla, CVPR 2018),
제약 최적화/라그랑주 방식 1편. 각각에 대해 `docs/paper/refs-audit.md` 에
**"우리와 다른 점 한 줄"** 을 적는다.

- [ ] **Step 4: 검증 — 모든 항목이 실재하는지 확인**

```bash
grep -c '^@' paper/refs.bib          # 기대: 25 이상
```
그리고 `docs/paper/refs-audit.md` 에 항목별로 `확인함 / 미확인` 을 적는다.
**미확인 항목은 본문에서 인용하지 않는다.** 지어낸 인용은 공백보다 나쁘다.

- [ ] **Step 5: 커밋**

```bash
git add paper/refs.bib docs/paper/refs-audit.md
git commit -m "paper: add verified bibliography with per-entry audit"
```

---

## Phase B — 한국어 개요

개요는 전부 `docs/paper/outline-ko.md` 한 파일에 절 순서대로 쌓는다.
각 Task 는 자기 절만 덧붙이고, 앞 절을 고치지 않는다.

### Task 3: §1 Introduction 개요

**Files:**
- Create: `docs/paper/outline-ko.md`
- Read: `docs/superpowers/specs/2026-09-14-eip-paper-spec.md`, `docs/paper/novelty-gate.md`

**Interfaces:**
- Consumes: `CONTRIB-1/2/3`, 증거 대장 A1~A8
- Produces: `## §1` 절 — 문단 P1~P6 각각에 [주장] [근거 수치] [인용] [의존 게이트] 네 줄

- [ ] **Step 1: 여섯 문단 골격을 아래 내용 그대로 적는다**

```markdown
## §1 Introduction

### P1 — SNN 의 가치 명제와 그 검증되지 않은 전제
[주장] SNN 의 효율은 "적게 발화한다" 는 전제 위에 서 있다. 그런데 정확도만 보고 학습한
       SNN 이 실제로 얼마나 발화하는지는 거의 보고되지 않는다.
[근거] ResNet-19 / CIFAR-10 / T=4 에서 규제 없는 학습의 결과는 486,656 spikes (n=5).
[인용] wu2018spatio, neftci2019surrogate, zheng2021going — 표준 직접 학습 설정
[게이트] 없음

### P2 — 그 발화의 절반은 필요 없다
[주장] 같은 학습 설정에서 스파이크를 51.5% 줄여도 정확도가 유지된다.
       즉 직접 학습된 SNN 은 과잉 발화 상태로 수렴한다.
[근거] baseline 96.654 ± 0.065 @ 486,656 (n=5)  vs  제안법 ρ=3e-3 96.650 @ 235,802 (n=4).
       차이 −0.004%p — baseline sd 의 1/16.
[인용] 없음 (본 연구 결과)
[게이트] 없음. **단 baseline 5런의 정본 목록(스펙 A1)을 쓸 것.** 글롭으로 모으면 2런만 잡힌다.

### P3 — 기존 계보와 그 두 가지 한계
[주장] 스파이크 총량을 벌하는 규제는 2016년부터 있었다. 남은 문제는 둘이다 —
       (a) 규제 세기를 손으로 맞춰야 하고 (b) 시뮬레이션 지표로만 보고한다.
[근거] Sorbaro 2020 은 α=1/S0² 로 정규화하되 **절대 SynOps 목표 S0 를 30개 모델로 스윕한다**
       (원문 156~158행, 381행). 1/S0² 은 학습 전에 확정되는 정적 상수다.
       Activity Pruning (NeurIPS 2025) 은 `eta`/`eta2` 를 설정 파일에 손으로 박는데,
       CIFAR-10 과 ImageNet-100 사이에서 `eta2` 가 100배, `eta` 가 10배 **서로 반대 방향**으로
       움직인다 (`cifar10.yaml` 1e-3/1e-4  vs  `imagenet100.yaml` 1e-4/1e-2).
       Sorbaro 는 Speck 제조사 그룹의 연구인데도 sinabs 시뮬 전용이다 (원문 확인).
[인용] neil2016learning, sorbaro2020optimizing, pellegrini2021low, perez2022surrogate,
       bu2025activity
[게이트] 없음 — 2026-09-14 원문·코드 확인 완료

### P4 — 곡선이 전부가 아니다 (본 논문의 핵심 관찰)
[주장] 가중치 설계를 19종 바꿔 봐도 전부 같은 정확도–스파이크 곡선에 떨어진다.
       그런데 곡선 위 같은 자리가 같은 내부 상태를 뜻하지 않는다.
[근거] 같은 스파이크 예산(≈193K)에서 plain L2 는 뉴런 155,669개, 제안법은 131,641개
       (−15.4%). 회귀 계수 0.869, t = −21.5, n = 21.
       뉴런당 발화: 규제없음 1.621 > 제안법 1.479 > 1−softmax 1.370 > plain L2 1.284
       (제안법 vs L2: t = 10.5, p = 8.4e-7).
       해석 — L2 는 모두를 약하게 만들어 줄이고, 제안법은 소수에 몰아준다.
[인용] 없음 (본 연구 결과)
[선점 처리] "스파이크 감소가 뉴런 침묵이다" 는 Activity Pruning (NeurIPS 2025) 이 이미 말했다.
       **이 문단에서 먼저 인정하고**, 우리가 더하는 것은 "그 침묵을 *어떤 뉴런에* 배분하는지가
       방법마다 다르다" 임을 분명히 한다.
[게이트] 없음

### P5 — 그래서 실기에서 무엇이 달라지는가
[주장] 뉴런을 적게 쓰는 것이 실제 뉴로모픽 하드웨어에서 이득으로 나타난다.
[근거] Speck 실측 — **아직 없음**
[인용] yao2024spikebased (Speck), horowitz2014computing (SynOps 회계 관행)
[게이트] **G7**. 아래 두 버전 중 하나를 쓴다.
         (7-있음) 실측 수치로 이득을 주장한다.
         (7-없음) 이 문단을 삭제하고, P4 뒤에 "뉴런 수 감소가 왜 이득인지" 를
                  G5(전체 테스트셋 영구 침묵 뉴런)로 대신 답한다.

### P6 — 기여
[주장] CONTRIB-1 / CONTRIB-2 / CONTRIB-3 을 불릿 3개로.
[게이트] CONTRIB-3 은 G7 이 비면 통째로 뺀다.
```

- [ ] **Step 2: P5 의 두 버전을 모두 작성한다**

G7 의 결과를 기다리지 않고 (7-있음)·(7-없음) 문단을 둘 다 써 둔다.
(7-없음) 버전은 G5 의 결과 자리(`__G5_SILENT_RATIO__`)를 남기되, **그 자리를 채울 명령을
같은 줄에 적는다**: `python extract_neuron_spikes.py --split test --n 10000`.

- [ ] **Step 3: 검증 — 수치 대조**

개요에 등장하는 모든 숫자를 스펙 증거 대장과 한 줄씩 대조한다.

```bash
grep -oE '[0-9]{1,3}(,[0-9]{3})+|[0-9]+\.[0-9]+' docs/paper/outline-ko.md | sort -u
```
출력의 각 값이 스펙 §2 에 있는지 확인한다. 없는 값이 하나라도 있으면 그 문장을 지운다.

- [ ] **Step 4: 커밋**

```bash
git add docs/paper/outline-ko.md
git commit -m "paper: outline introduction with evidence and gate annotations"
```

---

### Task 4: §2 Related Work 개요

**Files:**
- Modify: `docs/paper/outline-ko.md` (`## §2` 절 추가)
- Read: `paper/refs.bib`, `docs/paper/novelty-gate.md`

**Interfaces:**
- Consumes: Task 2 의 BibTeX 키, Task 1 의 겹침 판정
- Produces: `## §2` — 소절 6개, 각 소절에 [무엇을 다루는가] [인용 키 목록] [우리와의 차이 한 줄]

- [ ] **Step 1: 소절 6개를 아래 배치로 적는다**

```markdown
## §2 Related Work

### 2.1 Direct training of deep SNNs
[다루는 것] 대리 기울기 기반 직접 학습과 우리 실험 설정의 근거.
[인용] wu2018spatio, neftci2019surrogate, wu2019direct, zheng2021going, deng2022temporal,
       fang2021incorporating
[차이] 우리는 새 학습법을 제안하지 않는다. ResNet-19 + tdBN + T=4 는 그대로 물려받는다.
[분량] 2단 기준 6~8줄

### 2.2 Spike-count regularisation
[다루는 것] 손실에 스파이크 항을 더해 발화를 줄이는 계보. **우리 비교군의 문헌 근거.**
[인용] neil2016learning, sorbaro2020optimizing, pellegrini2021low, perez2022surrogate,
       sakemi2023sparse
[차이] 이들은 세기 계수를 손으로 정하고, 무엇을 벌할지(총량/SynOps)만 바꾼다.
       우리 실험은 **무엇을 벌하든 곡선이 같다**는 것을 보이고, 대신 곡선 위 같은 자리에서
       내부 상태가 다르다는 축을 연다.
[분량] 10~12줄 — 이 논문에서 가장 중요한 소절이다

### 2.3 Pruning and activity-based sparsification
[다루는 것] 선점 대응. 우리 기전 분석이 "이건 사실상 활동 기반 가지치기다" 로 나왔으므로
           정면으로 다룬다.
[인용] bu2025activity, li2024spiking, yao2023inherent
[차이] **겹치는 부분을 먼저 인정한다** — Activity Pruning 은 학습 중 스파이크 수 규제이고
       막전위(역치 적응)를 쓰며, "스파이크 감소는 뉴런 침묵" 과 "기존 활동 규제의 구조적 한계"
       를 이미 말했다. 우리가 더하는 것은 (1) 가중 방식 간 등가성 (2) 같은 예산에서의 뉴런
       사용 차이 (3) 세기 계수를 손으로 안 정한다는 것 셋이다.
       **인용거리:** 그쪽 설정 파일의 `eta2` 가 CIFAR-10 1e-4 → ImageNet-100 1e-2 로 100배,
       `eta` 는 1e-3 → 1e-4 로 10배 반대 방향으로 움직인다. 우리 문제 제기의 직접 근거다.
[분량] 10~12줄 — 선점 대응이라 짧으면 오히려 의심받는다

### 2.4 Balancing auxiliary loss terms
[다루는 것] 보조 손실의 가중치를 자동으로 정하는 계보 (SNN 밖).
[인용] GradNorm, Kendall uncertainty weighting, 제약 최적화 1편 — Task 2 Step 3 에서 확정
[차이] 이들은 여러 과제 손실 사이의 균형을 맞춘다. 우리는 과제 손실과 규제항 사이의
       비율을 고정한다.
[게이트] **G4** — `l2 + loss-ratio` 대조군이 없으면 이 소절을 §2 에서 빼고 §5 논의로 내린다.
[분량] 5~6줄

### 2.5 Energy accounting and neuromorphic deployment
[다루는 것] SynOps×pJ 회계 관행과 실기 배치.
[인용] horowitz2014computing, lemaire2022analytical, yao2024spikebased
[차이] 우리는 이 회계식을 **비판 대상이 아니라 비교 기준으로** 쓴다. 표준 계산식을 그대로
       쓰고 그 옆에 실측을 나란히 놓는다.
[게이트] G7 이 비면 "실측" 부분을 빼고 회계 관행 설명만 남긴다.
[분량] 6~8줄

### 2.6 Lateral inhibition and WTA in SNNs
[다루는 것] 이 연구의 **출발 동기**였던 계보.
[인용] cheng2020lisnn
[차이] **결과로 주장하지 않는다.** 우리 측정에서 억제상관은 +0.61~0.69 로 균일 대조군
       (+0.642)과 구별되지 않았다. 이 소절은 설계 의도 서술에만 쓰고, §5 에서 "의도한
       WTA 는 생기지 않았다" 를 명시한다.
[분량] 4~5줄. 넘치면 §1 P1 으로 흡수하고 소절을 없앤다.
```

- [ ] **Step 2: 분량 검산**

소절 분량 합이 2단 기준 40~50줄(약 3/4 쪽)인지 센다. 8쪽 논문에서 Related Work 가
1쪽을 넘으면 §4 결과가 밀린다. 넘으면 2.6 → 삭제, 2.4 → §5 이동 순으로 줄인다.

- [ ] **Step 3: 검증 — 인용 키가 실재하는지**

```bash
grep -oE '\b[a-z]+[0-9]{4}[a-z]+\b' docs/paper/outline-ko.md | sort -u > /tmp/keys.txt
grep -oE '^@\w+\{([^,]+),' paper/refs.bib | sed 's/.*{//;s/,//' | sort -u > /tmp/bib.txt
comm -23 /tmp/keys.txt /tmp/bib.txt
```
출력이 비어야 한다. 비지 않으면 refs.bib 에 없는 키를 인용한 것이다.

- [ ] **Step 4: 커밋**

```bash
git add docs/paper/outline-ko.md
git commit -m "paper: outline related work with six subsections and citation keys"
```

---

### Task 5: §4 실험 설정 개요 + 표·그림 목록 확정

**Files:**
- Modify: `docs/paper/outline-ko.md` (`## §4.1` 추가)
- Create: `docs/paper/figures-tables.md`

**Interfaces:**
- Consumes: 스펙 §1 실험 범위, §5 재현 경로
- Produces: `docs/paper/figures-tables.md` — T1~T6 / F1~F6 의 **번호·제목·생성 명령·의존 게이트**. Task 6·11·12·13 이 이 번호로만 표/그림을 부른다.

- [ ] **Step 1: 실험 설정 개요를 적는다**

```markdown
## §4.1 Experimental setup
- 모델: ResNet-19 (tdBN), Spikformer, SDT-V3 173M    ← VGG-16 제외
- 데이터: CIFAR-10, CIFAR-100, CIFAR10-DVS
- T = 4, 310 epochs, AdamW, cosine LR, wd 2e-2, cutmix (ep200 에서 종료)
- 비교군: 규제 없음 / plain L2 / 1−softmax / (G4 시 l2+loss-ratio)
- 보고: 최고 val_acc 시점의 행. 조건당 5런 → 사전 등록 규칙으로 4런 선별, n=4 평균.
  전 런 기록(돌린 개수·선별된 것·배제 사유)은 부록 표 A1.
- 스파이크 수는 eval/test 프로토콜. 학습 모드 S30/S1 은 선별에만 쓴다.
```

- [ ] **Step 2: 표 목록을 스펙 §8 에서 옮겨 적는다**

**번호를 새로 만들지 않는다.** 이전 세션에서 T1~T4·T7 과 F2/F3 이 이미 쓰였고, 스펙 §8 이
그 번호를 정본으로 고정했다. `docs/paper/figures-tables.md` 에 스펙 §8 의 표 8개(T1~T8)를
그대로 옮기고, **본문/부록 배치만 정한다.**

본문 최소 구성은 스펙이 못 박았다: **T1 · T3 · T4 · F1 · F4 · (F2+F3 합쳐 2패널)**.
8쪽에 더 들어갈 자리가 있으면 T2 → T8 순으로 본문에 올리고, 없으면 부록으로 내린다.
배치 판단의 근거를 한 줄씩 적는다.

- [ ] **Step 3: 그림 목록을 스펙 §8 에서 옮겨 적는다 (F1~F7)**

F2·F3·F4 항목에는 **캡션 필수 단서 두 개**를 함께 적는다 — (1) 후반층에서는 세 방법이
구분되지 않는다 (2) `dead_neuron_ratio` 는 배치 100장 기준이다.

그리고 스펙 §8 의 경고를 옮긴다: **층별 지니·상위10% 값이 두 세트 존재한다**
(09-10 측정 vs 09-14 `neuron_usage.py`). 한 논문 안에서 섞지 말고 하나를 골라 전부
그것으로 쓴다. **어느 세트를 쓸지 여기서 정하고 근거를 적는다.**

- [ ] **Step 4: Method 용어 대응표를 만든다**

`docs/paper/method-terms.md` — Method 절이 바뀌어도 다른 절을 고치지 않기 위한 간접층이다.
§4.3 어블레이션 표의 행 이름은 **반드시 이 표의 오른쪽 열만** 쓴다.

```markdown
| 코드 플래그 (run_paper.py) | 논문 표기 | 본문에서 부르는 법 |
|---|---|---|
| prop (전체)                | Ours            | the proposed regulariser |
| abl_novmem                 | w/o potential   | the sub-threshold term |
| abl_noinv                  | w/o inversion   | the inversion |
| abl_nofinal                | w/o final-step  | the single-step application |
| abl_nolr                   | fixed $\lambda$ | the ratio controller |
| l2                         | Plain L2        | the layer-wise $L_2$ penalty |
| sm                         | 1-softmax       | the softmax weighting |
```

Method 가 확정되면 **이 파일만** 고친다. `paper/sections/*.tex` 는 건드리지 않는다.

- [ ] **Step 5: 검증 — 생성 명령이 실제로 도는지, 번호가 스펙과 일치하는지**

```bash
grep -oE '\bT[1-9]\b|\bF[1-7]\b' docs/paper/figures-tables.md | sort -u
grep -oE '\bT[1-9]\b|\bF[1-7]\b' docs/superpowers/specs/2026-09-14-eip-paper-spec.md | sort -u
```
두 출력이 같아야 한다. 다르면 번호를 새로 만든 것이다.


```bash
cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py && \
  /home/kyccj/anaconda3/envs/venv_1/bin/python lambda_epoch.py && ls -la *.png
```
두 스크립트가 오류 없이 png 를 갱신해야 한다. F1·T5 는 아직 스크립트가 없으므로
`docs/paper/figures-tables.md` 에 **"스크립트 없음 — Task 13 에서 작성"** 으로 표시한다.

- [ ] **Step 6: 커밋**

```bash
git add docs/paper/outline-ko.md docs/paper/figures-tables.md docs/paper/method-terms.md
git commit -m "paper: fix table and figure inventory with generation commands"
```

---

### Task 6: §4 결과·분석, §5 논의, §6 결론 개요

**Files:**
- Modify: `docs/paper/outline-ko.md`

**Interfaces:**
- Consumes: `docs/paper/figures-tables.md` 의 T/F 번호, 스펙 증거 대장
- Produces: `## §4.2`~`## §6` — 각 소절에 [주장 한 줄] [받치는 T/F] [게이트]

- [ ] **Step 1: §4.2 주 결과**

```markdown
### §4.2 Main result — 무손실 51% 감소
[주장] 정확도를 잃지 않고 스파이크를 51.5% 줄인다.
[표] T1.  baseline 96.654 ± 0.065 @ 486,656 (n=5)  vs  제안법 ρ=3e-3 96.650 @ 235,802 (n=4)
[그림] F1
[게이트] 없음
[쓰지 않을 것] plain L2 대비 정확도 우위. +0.071%p, t=1.41 이다. 같은 자리의 L2 는
              96.550 @ 246,865 이고 이 차이는 주장할 수 없다.
```

- [ ] **Step 2: §4.3 어블레이션**

```markdown
### §4.3 Ablation
[주장] 두 축이 서로 다른 요소를 짚는다 — 정확도는 하나에서, 집중은 다른 하나에서 나온다.
[표] T3
[확정된 것] vmem 항: 곡선 대비 −0.222, 3.5σ, n=4. 네 요소 중 유일하게 확정된 정확도 기여.
[크기로 확정된 것] `1−` 반전: 같은 ρ=4e-3 에서 착지점 192K → 437,011. 규제 없음(481K)과
                  거의 같다. σ 로 잴 필요 없다.
[흐린 것] −final_step (n=2, SE 0.188), −loss-ratio (n=2). **G2 전에는 표에 σ 를 적지 않는다.**
```

- [ ] **Step 3: §4.4 일반성**

```markdown
### §4.4 Generality
[주장] 같은 ρ 가 데이터셋·아키텍처를 건너 전이된다.
[표] T4
[게이트] G3(C100 n≥3), G6(Spikformer·DVS 재실행), G8(SDT-V3).
[현재 상태] C100 은 조건당 n=1 이라 순서를 못 정한다. l2 5e-7 이 80.660, 제안법 3e-3 이
           80.370 인데 착지점이 달라 곡선도 안 그려진다. **G3 전에는 이 소절을 쓰지 않는다.**
```

- [ ] **Step 4: §4.5 분석 — 무엇이 달라졌는가**

```markdown
### §4.5 What changes inside the network
[주장] 스파이크 감소는 거의 전부 "뉴런이 꺼져서" 일어나고, 방법에 따라 **어떤 뉴런을 끄는지**가 다르다.
[그림] F2, F3, F4
[수치] 같은 예산 193K 에서 활성 뉴런 155,669 (L2) vs 131,641 (제안법), −15.4%, t=−21.5, n=21
       뉴런당 발화 1.284 (L2) vs 1.479 (제안법), t=10.5, p=8.4e-7
       conv2_block1 지니 0.695 → 0.758 (L2) → 0.894 (제안법), 상위10% 점유 0.805
[반드시 쓸 한계] **conv3_block2 이후 층에서는 세 방법이 구분되지 않는다.**
               집중은 스파이크가 많은 초·중반층에서만 일어난다.
[반드시 쓸 한계] `dead_neuron_ratio` 는 배치 100장 기준이다. "영구히 죽은 뉴런" 으로
               읽으면 안 된다. **G5** 가 채워지면 테스트셋 10,000장 기준으로 바꾼다.
```

- [ ] **Step 5: §5 논의 + Limitations**

```markdown
### §5 Discussion
- 곡선이 두 방법을 구분하지 못하지만 뉴런 사용은 구분한다 — 규제 비교의 기준을 곡선 하나로
  두는 관행에 대한 반례.
- 의도했던 WTA 는 생기지 않았다. 억제상관 +0.61~0.69, 균일 대조군 +0.642 와 구별 안 됨.
  **이걸 숨기지 않고 쓴다.** §2.6 과 짝을 이룬다.
- 학습 비용: **규제를 켜면 baseline 631 → 695 ms/step (+10.0%). 이는 모든 비교군보다 적다**
  (plain L2 +12.8%, 1−softmax +20.2%, 매 시점 호출 +33.2%). T4 참조.
  "제안법이 L2 보다 싸다" 만 쓰면 baseline 대비 비용을 숨기는 것이 된다.
- (G4 시) λ 자동화 — loss-ratio 는 plain L2 에도 붙는다. `l2+loss-ratio` 대조군 결과로만 말한다.
- **SynOps 축은 빼고 "같은 SynOps 에서 뉴런을 적게 쓴다" 로 쓴다** (스펙 §9-D).
  스파이크당 SynOps 가 1,505 vs 1,488 = +1.1% 라 에너지 표로는 우위가 안 나온다.
- **고지 의무 §9-A~I 를 Limitations 또는 본문 해당 위치에 배치한다.** 특히 §9-B(λ ep201
  급락 원인 미확인)는 **F6 과 CONTRIB-2 를 쓰기 전에 확인**해야 한다 — 확인 전에는
  "스케줄과 무관하다" 를 주장하지 않는다.

### §5.x Limitations
- 조건당 n=4 이고, 조합별로는 유의하지 않다. 주장은 방향의 일관성에 근거한다.
- plain L2 대비 정확도 우위를 주장하지 않는다 (t=1.41, p<0.05 까지 60런 필요).
- 집중은 초·중반층 현상이다.
- (G7 없으면) 실기 검증이 없다는 것을 명시한다.
```

- [ ] **Step 6: §6 Conclusion 개요 — 세 문장**

`§1 P6` 의 CONTRIB-1/2/3 을 과거형으로 다시 쓰고, 마지막 문장은 향후 과제(실기 배포 또는
대형 모델)로 연결한다. 새 주장을 여기서 처음 꺼내지 않는다.

- [ ] **Step 7: 검증 — 주장–증거 대조표를 만든다**

`docs/paper/claim-evidence.md` 에 표를 만든다: 개요의 모든 [주장] 줄 × 받치는 T/F 번호 ×
스펙 등급(A/B/C) × 게이트. **등급 C 인데 게이트가 안 적힌 행이 있으면 실패다.**

- [ ] **Step 8: 커밋**

```bash
git add docs/paper/outline-ko.md docs/paper/claim-evidence.md
git commit -m "paper: outline results, discussion and conclusion with claim-evidence map"
```

---

### Task 7: 개요 전체 검토 게이트

**Files:**
- Create: `docs/paper/outline-review.md`

**Interfaces:**
- Consumes: `docs/paper/outline-ko.md` 전체, `docs/paper/claim-evidence.md`
- Produces: 사용자 승인 기록. **승인 없이 Phase C 로 넘어가지 않는다.**

- [ ] **Step 1: 세 가지를 기계적으로 검사한다**

1. 개요의 모든 숫자가 스펙 증거 대장에 있는가 (Task 3 Step 3 의 grep 을 전체 파일에)
2. 모든 인용 키가 `paper/refs.bib` 에 있는가 (Task 4 Step 3 의 comm)
3. 기각된 주장 셋이 어디에도 없는가:
```bash
grep -nE 'WTA 가 (일어|생기)|정확도.*우위|전 층' docs/paper/outline-ko.md
```
출력이 있으면 그 줄을 고친다 (§2.6·§5 의 "생기지 않았다" 서술은 예외로 허용).

- [ ] **Step 2: Method 누설 검사**

```bash
grep -nE '1-maxnorm|maxnorm|final_step|vmem|loss-ratio|wta_rev|sc_rate' docs/paper/outline-ko.md
```
§4.3 어블레이션 행 이름과 §5 논의의 loss-ratio 문장 외에 나오면 Method 를 전제한 것이다.
일반 명사("제안법의 한 구성요소")로 바꾼다.

- [ ] **Step 3: 분량 검산**

절별 목표: §1 1쪽 / §2 0.75쪽 / §3 Method 1.5쪽(비움) / §4 3쪽 / §5 1쪽 / §6 0.25쪽.
개요의 문단 수 × 예상 줄 수로 검산해 `docs/paper/outline-review.md` 에 적는다.

- [ ] **Step 4: 사용자에게 개요 전체를 제시하고 승인받는다**

승인 항목 셋을 명시적으로 묻는다: (a) §1 의 논지 순서 (b) §2 소절 구성과 분량
(c) 기각한 주장 셋에 동의하는지.

- [ ] **Step 5: 커밋**

```bash
git add docs/paper/outline-review.md
git commit -m "paper: record outline review and approval gate"
```

---

## Phase C — 영문 LaTeX 초고

### Task 8: paper/ 스캐폴딩

**Files:**
- Create: `paper/main.tex`, `paper/sections/{intro,related,method,experiments,discussion,conclusion}.tex`
- Create: `paper/Makefile`
- Create: `paper/.gitignore`

**Interfaces:**
- Consumes: `paper/refs.bib`
- Produces: `make -C paper` 가 `paper/main.pdf` 를 만드는 상태. Task 9~14 가 `sections/*.tex` 만 고친다.

- [ ] **Step 1: CVPR 스타일 파일을 받아 `paper/` 에 둔다**

공식 배포본(`cvpr.sty`, `ieee_fullname.bst`)을 받는다. 받을 수 없으면
`\documentclass[twocolumn,letterpaper,10pt]{article}` + `geometry` 로 임시 조판하고,
`paper/README.md` 에 "스타일 파일 교체 필요" 를 적는다.

- [ ] **Step 2: `main.tex` 를 만든다**

```latex
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage{cvpr}
\usepackage{graphicx,booktabs,amsmath,amssymb}
\usepackage[pagebackref,breaklinks,colorlinks]{hyperref}
\begin{document}
\title{TBD}
\maketitle
\begin{abstract}
\input{sections/abstract}
\end{abstract}
\input{sections/intro}
\input{sections/related}
\input{sections/method}
\input{sections/experiments}
\input{sections/discussion}
\input{sections/conclusion}
{\small\bibliographystyle{ieee_fullname}\bibliography{refs}}
\end{document}
```

- [ ] **Step 3: `sections/method.tex` 를 자리표시자로 만든다**

```latex
\section{Method}
\label{sec:method}
% 이 절은 성능 확정 후 작성한다. 2026-09-14 기준 의도적 공백.
% 다른 절은 이 절의 내부 구조를 전제하지 않는다 - 규약은 docs/paper/method-terms.md.
\textit{[Method section pending.]}
```

- [ ] **Step 4: `Makefile`**

```make
main.pdf: main.tex $(wildcard sections/*.tex) refs.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf
```

- [ ] **Step 5: 검증 — 빈 문서가 빌드되는지**

```bash
make -C paper && ls -la paper/main.pdf
```
PDF 가 생겨야 한다.

- [ ] **Step 6: 커밋**

```bash
git add paper/ && git commit -m "paper: scaffold CVPR-format LaTeX skeleton with empty method section"
```

---

### Task 9: §1 Introduction 영문

**Files:**
- Modify: `paper/sections/intro.tex`
- Read: `docs/paper/outline-ko.md` `## §1`

**Interfaces:**
- Consumes: 개요 P1~P6, `paper/refs.bib` 키
- Produces: `\label{sec:intro}`, 기여 불릿 3개 (`\item` 3개)

- [ ] **Step 1: P1~P6 을 문단 6개로 옮긴다**

개요의 [근거] 수치를 본문에 그대로 넣는다. 반올림하지 않는다 (96.650, 235{,}802).
게이트가 걸린 문장은 `% GATE: G7` 처럼 해당 게이트 번호를 주석으로 바로 위 줄에 단다.

- [ ] **Step 2: 기여 불릿을 쓴다**

```latex
\begin{itemize}\itemsep0pt
  \item CONTRIB-1 ...
  \item CONTRIB-2 ...
  \item CONTRIB-3 ...   % GATE: G7 - drop if no hardware measurement
\end{itemize}
```

- [ ] **Step 3: 검증 — 빌드와 미결 인용 확인**

```bash
make -C paper 2>&1 | grep -iE 'undefined|warning.*citation' ; ls paper/main.pdf
```
`Citation ... undefined` 가 없어야 한다.

- [ ] **Step 4: 커밋**

```bash
git add paper/sections/intro.tex && git commit -m "paper: draft introduction"
```

---

### Task 10: §2 Related Work 영문

**Files:**
- Modify: `paper/sections/related.tex`

**Interfaces:**
- Consumes: 개요 `## §2` 소절 6개
- Produces: `\subsection` 6개 (또는 분량 검산 후 줄인 개수)

- [ ] **Step 1: 소절을 옮긴다.** 각 소절 마지막 문장이 반드시 "우리와의 차이" 여야 한다.
- [ ] **Step 2: 2.3 에서 선점을 먼저 인정하고 남는 것을 말하는 순서로 쓴다.** 방어적으로 쓰면 리뷰어가 더 판다.
- [ ] **Step 3: 검증**

```bash
make -C paper 2>&1 | grep -ic 'citation.*undefined'    # 기대: 0
grep -c subsection paper/sections/related.tex
```

- [ ] **Step 4: 커밋**

```bash
git add paper/sections/related.tex && git commit -m "paper: draft related work"
```

---

### Task 11: §4 실험 — 설정과 주 결과

**Files:**
- Modify: `paper/sections/experiments.tex`
- Create: `paper/tables/gen_tables.py`

**Interfaces:**
- Consumes: `collect_paper.py` 출력, `docs/paper/figures-tables.md` 의 T1
- Produces: `paper/tables/t1_main.tex` (booktabs). Task 12·13 이 같은 스크립트에 표를 덧붙인다.

- [ ] **Step 1: 표 생성 스크립트를 쓴다**

`paper/tables/gen_tables.py` 는 `collect_paper.py` 의 CSV 를 읽어 `.tex` 조각을 낸다.
열 순서는 **train loss, train acc, val loss, val acc, spikes** 로 고정한다.
반복 런은 개별 행을 모두 내고 **그룹 바로 아래에 평균 행**을 넣는다 (`\midrule` 없이 `\cmidrule`).

- [ ] **Step 2: T1 을 생성한다**

**먼저 `docs/paper/runs-t1.txt` 에 런 디렉토리를 한 줄씩 명시한다.** baseline 5런이
다섯 디렉토리에 흩어져 있어 `'_paper/*'` 글롭은 2런만 잡는다 (스펙 A1).

```
_paper/r19c10-base---s11
_paper/r19c10-base---s12
_baseline_more/base_r19_c10_run3
_baseline_no_reg_r19_c10
_baseline_trajectory
_paper/r19c10-prop-3e-3-s2
... (제안법·L2·1−softmax 의 선별된 런)
```

```bash
cd /home/kyccj/PycharmProjects/TensorFlow-SNNs && \
/home/kyccj/anaconda3/envs/venv_1/bin/python collect_paper.py \
  $(grep -v '^#' docs/paper/runs-t1.txt | tr '\n' ' ') --out _results/paper.csv && \
/home/kyccj/anaconda3/envs/venv_1/bin/python paper/tables/gen_tables.py --in _results/paper.csv --table t1
```

**검산:** baseline 행이 5개, 평균이 96.654 @ 486,656 이어야 한다. 2개만 나오면 목록이 빈 것이다.

- [ ] **Step 3: §4.1 설정과 §4.2 주 결과 본문을 쓴다.** 개요 Step 1·2 를 옮긴다.
- [ ] **Step 4: 검증 — 표의 수와 CSV 의 수가 같은지**

```bash
grep -oE '9[0-9]\.[0-9]+' paper/tables/t1_main.tex | sort -u
```
출력값이 `_results/paper.csv` 의 `best_val_acc` 에 모두 존재해야 한다.

- [ ] **Step 5: 커밋**

```bash
git add paper/sections/experiments.tex paper/tables/ && \
git commit -m "paper: add setup and main result with generated table"
```

---

### Task 12: §4.3~4.4 어블레이션과 일반성

**Files:**
- Modify: `paper/sections/experiments.tex`, `paper/tables/gen_tables.py`

**Interfaces:**
- Consumes: T3, T4
- Produces: `paper/tables/t3_ablation.tex`, `paper/tables/t4_generality.tex`

- [ ] **Step 1: T3 를 3열 구조로 만든다** — 정확도(곡선 대비), 착지점(스파이크), 뉴런당 발화.
- [ ] **Step 2: n<4 인 팔은 σ 를 비우고 각주로 "n=2" 를 단다.** 값을 지어내지 않는다.
- [ ] **Step 3: §4.4 는 G3·G6·G8 이 모두 비면 `% PENDING` 주석만 남기고 본문을 쓰지 않는다.**
- [ ] **Step 4: 검증**

```bash
make -C paper && grep -c 'n{=}2' paper/sections/experiments.tex
```

- [ ] **Step 5: 커밋**

```bash
git add -A paper && git commit -m "paper: add ablation and generality sections"
```

---

### Task 13: §4.5 분석 절과 그림

**Files:**
- Modify: `paper/sections/experiments.tex`
- Create: `EIP_fig/pareto_curve.py` (F1), `paper/figures/` (png 복사본)

**Interfaces:**
- Consumes: `docs/paper/figures-tables.md` 의 F1~F5, `EIP_fig/neuron_usage.py` (F2~F4), `EIP_fig/lambda_epoch.py` (F5)
- Produces: F1 스크립트와 `paper/figures/*.pdf`

- [ ] **Step 1: F1 을 쓴다** — 가중 설계 19종의 정확도–스파이크 산점도와 단일 적합 곡선.
**축은 실제 정확도와 실제 스파이크 수다.**

**데이터 위치 (2026-09-14 확인):** `/media/hdd1/kyccj/EIP/archive` 에는 571개 디렉토리가 있지만
**`train.log` 가 있는 것은 15개뿐**이다. 이전 때 체크포인트와 `reg_detail.csv`(423개)만 옮겼고
`train.log` 는 저장소에 남았다. 정확도·스파이크는 `train.log` 에서 나오므로 **저장소 쪽을 본다**:

```bash
find . -maxdepth 3 -name train.log | wc -l     # 100개
```
`_sweep_wta_rev*`, `_rho_*`, `_grp_*`, `_vm*`, `_chwise`, `_pnorm`, `_maxnorm*` 계열이
가중 설계 19종에 해당한다. 방법별 대표점을 고르는 기준(어떤 λ 를 대표로 쓸지)을
`docs/paper/figures-tables.md` 에 먼저 적고 그림을 그린다.

- [ ] **Step 2: 그림을 PDF 로 내보낸다** (LaTeX 는 벡터가 낫다). `plt.savefig(..., format='pdf')`.
- [ ] **Step 3: §4.5 본문을 쓴다.** 개요 Step 4 의 두 한계 문장을 **반드시** 포함한다.
- [ ] **Step 4: 검증**

```bash
cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python pareto_curve.py && \
ls -la /home/kyccj/PycharmProjects/TensorFlow-SNNs/paper/figures/
make -C /home/kyccj/PycharmProjects/TensorFlow-SNNs/paper
```

- [ ] **Step 5: 커밋**

```bash
git add -A paper EIP_fig && git commit -m "paper: add internal-state analysis section and figures"
```

---

### Task 14: §5 논의·한계, §6 결론, 초록

**Files:**
- Modify: `paper/sections/discussion.tex`, `paper/sections/conclusion.tex`
- Create: `paper/sections/abstract.tex`

**Interfaces:**
- Consumes: 개요 §5·§6, 확정된 CONTRIB-1/2/3
- Produces: 완성된 본문

- [ ] **Step 1: §5 를 쓴다.** WTA 가 생기지 않았다는 서술을 **빼지 않는다.**
- [ ] **Step 2: Limitations 를 독립 소절로 둔다.** 개요의 네 항목 + **스펙 §9 의 고지 의무
중 §4·§5 에 자리가 없는 것 전부.** §9-A(도함수 아님)·§9-E(대리 기울기 불일치)·
§9-F(QKFormer 포팅 버그)·§9-G(eval 프로토콜)·§9-H(두 종류의 침묵 뉴런)는 빠뜨리면
리뷰어가 잡는다.
- [ ] **Step 3: §6 은 세 문장.** 새 주장 금지.
- [ ] **Step 4: 초록을 마지막에 쓴다.** 150~200 단어. 구성: 문제 1문장 → 관찰 1문장 →
      핵심 수치 2문장(51.5% 감소 / 같은 예산에서 −15.4% 뉴런) → 기여 1문장 → 검증 범위 1문장.
- [ ] **Step 5: 검증**

```bash
make -C paper && pdftotext paper/main.pdf - | head -40
wc -w < paper/sections/abstract.tex     # 기대: 150~220
```

- [ ] **Step 6: 커밋**

```bash
git add -A paper && git commit -m "paper: add discussion, limitations, conclusion and abstract"
```

---

### Task 15: 전체 자체 검토

**Files:**
- Create: `docs/paper/draft-review.md`

**Interfaces:**
- Consumes: `paper/main.pdf`, `docs/paper/claim-evidence.md`
- Produces: 검토 기록과 남은 게이트 목록

- [ ] **Step 1: 수치 대조.** PDF 의 모든 숫자를 스펙 증거 대장과 대조한다.

```bash
pdftotext paper/main.pdf - | grep -oE '[0-9]{1,3}(,[0-9]{3})+|[0-9]+\.[0-9]{2,}' | sort -u
```

- [ ] **Step 2: 인용 대조.**

```bash
make -C paper 2>&1 | grep -iE 'undefined|multiply defined'
```
출력이 없어야 한다.

- [ ] **Step 3: 분량.** 본문이 8쪽 이내인지. 넘으면 §2.6 → §4.4 순으로 줄인다.

```bash
pdfinfo paper/main.pdf | grep Pages
```

- [ ] **Step 4: 기각 주장 재검사.** Task 7 Step 1-3 의 grep 을 `paper/sections/*.tex` 에.
- [ ] **Step 5: Method 누설 재검사.** Task 7 Step 2 의 grep 을 `paper/sections/*.tex` 에.
- [ ] **Step 6: 남은 게이트를 목록으로 정리해 사용자에게 보고한다.**
- [ ] **Step 7: 커밋**

```bash
git add docs/paper/draft-review.md && git commit -m "paper: record draft self-review and remaining gates"
```

---

## Self-Review (계획 작성자 기록)

**스펙 커버리지** — 스펙 §2 의 A1~A8 은 §1 P2·P4 와 §4.2·§4.3·§4.5 가 받는다.
B1~B2 는 게이트 G2·G3 으로, C1~C5 는 G5·G7·G4·G6·G8 로 대응된다. G1(baseline 표본)은 2026-09-14 에 해소됐다.
스펙 §3(기각 주장)은 Task 7 Step 1-3 과 Task 15 Step 4 가 기계적으로 검사한다.
스펙 §6(보고 형식)은 Global Constraints 와 Task 11 Step 1 에 들어갔다.

**남은 공백** — 사용자의 헤드라인 세 번째 문장("Speck 으로 실기 검증")은 **G7 이 비어 있어
현재 어떤 Task 도 그 근거를 만들지 못한다.** Speck 코드는 이 레포에 없고 다른 머신에 있다.
Task 3 Step 2 가 두 버전의 인트로를 준비하는 것이 이 공백에 대한 대응이며,
근본 해결은 Phase 0 의 G7 을 별도 작업으로 띄우는 것이다.

**Method 의존** — §4.3 어블레이션만 Method 내부 구조를 전제한다. 이는 불가피하므로
`docs/paper/method-terms.md` 의 대응표를 통해 이름을 간접 참조하고, Method 가 바뀌면
그 파일 하나만 고치면 되도록 했다. 그 파일은 Task 5 Step 4 에서 만든다.
