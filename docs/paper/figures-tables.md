# 표·그림 인벤토리 (Task 5, 2026-09-14)

정본은 `docs/superpowers/specs/2026-09-14-eip-paper-spec.md` §8 이다. 이 문서는 그 번호를
그대로 옮기고, 8쪽 제약 아래 본문/부록 배치와 그 근거만 정한다. **번호를 새로 만들지 않는다.**
Task 6·11·12·13 은 이 문서의 번호로만 표/그림을 부른다.

본문 최소 구성: **T1 · T3 · T4 · F4 · (F2+F3, 2패널 합본)**. 스펙 §8 이 원래 못 박은
구성에는 F1(파레토 곡선, 가중 설계 등가성)이 있었으나, **2026-09-14 사용자 결정으로
F1 은 만들지 않는다** — 근거가 논문 범위 밖인 VGG16-CIFAR10 뿐이고(A8, 아래 "F1 데이터·
대표점" 절), 유효 설계 수도 09-14 이전 문서가 적던 19종이 아니라 13종이다. CONTRIB-1
은 A2(활성 뉴런 13.1%) 하나로 서고, 등가성 관찰은 그림·수치 없이 동기 서술로만 남는다.
나머지는 스펙 §8 이 제안한 배치를 기본값으로 하되, 8쪽 여유를 봐서 T2 → T8 순으로 본문에
올리고 안 되면 부록으로 내린다.

## 표 (T1–T8)

| ID | 내용 | 생성 명령 | 게이트 | 배치 | 근거 |
|---|---|---|---|---|---|
| T1 | R19-C10 주 비교표 — base/l2/sm/prop 이 190~202K 스파이크 구간에 모임 | `python collect_paper.py '_paper/r19c10-{base,l2,sm,prop}-*' --out _results/t1_r19c10_main.csv` 후 `docs/paper/runs-t1.txt` 의 정본 런 목록(baseline 유효 6런에 규칙 3 상위 4개, 다섯 디렉토리에 흩어져 있음)으로 수동 대체 — 글롭만 쓰면 일부만 잡혀 평균이 어긋난다 | 없음 | **본문 (필수)** | 헤드라인 C1(무손실 51.7% 감소)의 유일한 직접 증거. 스펙이 본문으로 못 박음 |
| T2 | 선행 연구 대비 (SOTA 표) | 자동화 없음 — 문헌값 수기 입력 (§2.2/§2.3 인용 논문의 보고 수치) | 없음 | 본문 | 표가 작고(수기 입력, 성능 축만) 독자가 §1 헤드라인을 문헌 맥락에 놓는 데 필요 — 8쪽 여유가 있는 한 유지 |
| T3 | 어블레이션 — 2단 구성(같은 ρ 행 + 스파이크 맞춘 행) | 같은 ρ 행: `python collect_paper.py '_paper/r19c10-abl_{novmem,nofinal,noinv,nolr}-*' --out _results/t3_ablation_samerho.csv`. 맞춘 행: 스펙 §12-4 목표(−vmem ρ≈2.6e-3, −final_step ρ≈2.9e-3, −loss-ratio λ≈3.2e-7)로 `run_paper.py` 재실행 후 동일 명령으로 재집계 | **맞춘 행 12런 미실행 — 승인 대기 (§12-4)** | **본문 (필수)** | 스펙이 본문으로 못 박음. 맞춘 행이 비어 있으면 같은 ρ 행만으로 축소해 싣고 "맞춘 행 미실행"을 표 각주에 명시 |
| T4 | 학습 비용 (ms/step) | 완료 — 스펙 §2 A7 표를 그대로 전재 (310에폭 평균, train.log 타임스탬프 기반 수기 계산). 자동 재생성 스크립트 없음 | 없음 | **본문 (필수)** | 스펙이 본문으로 못 박음. "규제 시 10% 느려지되 전 비교군 중 가장 쌈"이 CONTRIB-2 의 비용 근거 |
| T5 | 에너지 추정 (SynOps × pJ, Horowitz 계수) | 신규 스크립트 필요 (미작성) | **§9-D — 우위 주장 불가** | 부록 | §9-D: SynOps 축은 비교군 간 우위 근거가 안 된다(+1.1%, 차이 없음). baseline 대비 총량 감소 참고용으로만 부록에 싣는다 |
| T6 | Speck 실측 | 별도 머신(sinabs), 이 레포에 코드 없음 | **G7 — 미착수** | 본문 (있으면), 없으면 표 자체를 생략 | 스펙 §8 이 "본문 (있으면)"으로 지정. C2: ResNet-19/제안법으로 Speck 을 한 번도 돌린 적 없어 현재는 빈 자리 |
| T7 | 부록 — 배제된 런과 사유 (시드 고정 부작용 `_paper_bad_seeded/` 95.740 ± 0.418 포함) | `python collect_paper.py '_paper/*' '_paper_bad_seeded/*' --out _results/t7_excluded.csv --print` (exclude/exclude_reason 열 확인) | 없음 | 부록 | 사전 등록 배제 로그는 본문 서사에 필요 없고 재현성 부록에 속하는 내용 (§12-6 성격과 동일) |
| T8 | 데이터셋·아키텍처 일반성 (C100/Spikformer/DVS/SDT-V3) | `python collect_paper.py '_paper/r19c100-*' 'spikformer/_paper/*' 'dvs/_paper/*' --out _results/t8_generality.csv` (Spikformer/DVS 는 제안법 재실행 후) | **G3(C100 n=1)·G6(Spikformer/DVS 구 sweep)·G8(SDT-V3 미착수)** | 본문 (게이트 충족 시), 미충족 시 부록 축소판 또는 생략 | 스펙 §8 이 본문으로 지정했으나 2026-09-14 기준 세 게이트 모두 미충족(C100 n=1이라 곡선 불가, Spikformer/DVS 는 wta_rev 시절 sweep) — 제출 시점까지 못 채우면 "현재 진행 중" 각주와 함께 부록으로 내린다 |
| T9 | 활성 뉴런 축 전체 표 (09-13, 팔별 스파이크/활성뉴런/스파이크당뉴런) | 수기 집계 (스펙 §8 "층별 세부값" 아래 원자료 표를 그대로 전재) | 없음 — **아직 후보, 정식 채택 안 됨** | 부록 (**아직 후보, 정식 채택 안 됨**) | 스펙 §8 이 "표 T9 후보 — 부록" 으로만 표기하고 본문 배치를 지시하지 않음. F4(같은 예산에서 쓰는 뉴런 수)가 이미 본문에서 같은 축(스파이크당 뉴런)을 다루므로, 이 표는 팔별 전체 수치를 원할 때 참고하는 부록 보충 자료로만 둔다 |

## 그림 (F1–F7)

| ID | 내용 | 생성 명령 | 게이트 | 배치 | 근거 |
|---|---|---|---|---|---|
| ~~F1~~ | **삭제 (2026-09-14 사용자 결정).** 정확도–스파이크 곡선, 가중 설계 등가성. `EIP_fig/pareto_curve.py` 와 `paper/figures/f1_pareto_curve.pdf` 를 모두 지웠다 — 다시 만들지 않는다는 결정이라 생성기를 남겨 두지 않는다 | (삭제됨) | — | **없음 — 만들지 않는다** | 근거가 논문 범위 밖 VGG16-CIFAR10 뿐이고, Task 13 재검증 결과 유효 설계가 09-14 이전 문서의 19종이 아니라 13종. 등가성은 그림·수치 없는 동기 서술로만 §1 P4 에 남는다 (spec §2 A8). 상세 경위는 아래 "F1 데이터·대표점" 절 |
| F2 | 층별 지니 계수 (baseline/l2/prop) | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 c, `paper/figures/f2_f3_gini_top10.pdf` 로도 저장) | 없음 | **본문 (필수, F3 과 2패널 합본)** | 스펙이 F2+F3 합본을 본문 최소 구성으로 못 박음 |
| F3 | 층별 상위 10% 점유 | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 d, `paper/figures/f2_f3_gini_top10.pdf` 로도 저장) | 없음 | **본문 (필수, F2 와 2패널 합본)** | 위와 동일 |
| F4 | 같은 예산에서 쓰는 뉴런 수 — 회귀 두 선이 전 구간 평행 | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 a, `paper/figures/f4_active_neurons.pdf` 로도 저장) | 없음 | **본문 (필수)** | 스펙이 본문으로 못 박음. A2(활성 뉴런 13.1% 차이) 의 직접 증거, CONTRIB-1 의 핵심 그림 |
| F5 | 뉴런당 발화 강도 막대 (4방법) | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 b, `paper/figures/f5_intensity_appendix.pdf` 로도 저장, `paper/figures/f5_intensity.tex` 로 래핑) | 없음 | 부록 | A3(뉴런당 발화 강도 순서) 의 보조 증거 — 본문 서사(F1·F4·F2+F3)가 이미 "곡선 동일 / 뉴런 사용 차이" 를 전달하므로, 강도 축은 부록에서 수치로만 보강 |
| F6 | 에폭별 λ 궤적과 정확도 (ep200 cutmix 종료 반응) | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python lambda_epoch.py` (`paper/figures/f6_lambda_epoch_appendix.pdf` 로도 저장, `paper/figures/f6_lambda_epoch.tex` 로 래핑, `\input` 은 게이트로 주석 처리) | **§9-B — ep201 급락이 LR 스케줄과 겹치는지 미확인** | 부록 | **1차 사유(§9-B 사용 게이트)**: 스펙 §9-B는 "F6 과 CONTRIB-2 를 쓰기 전에 확인해야 한다. 확인 전에는 '스케줄과 무관하다' 를 주장하지 않는다" 고 명시한다 — 미확인 상태에서는 F6 사용 자체가 제한된다. **2차 사유(지면)**: 게이트가 풀리더라도 본문은 T1·T2·T3·T4·F1·F4·(F2+F3) 만으로 이미 8쪽이 찬다. **CONTRIB-2 의 본문 근거는 F6 이 아니라 T1/T3 수치다** — 스펙 §9-C 가 승인한 CONTRIB-2 문장은 "제안법은 고정 λ보다 중반 구간(ep51–200)에서 스파이크를 18.7~20.3% 더 줄이면서 정확도를 잃지 않는다" 라는 수치 비교이지 궤적 그림이 아니므로, F6 을 부록으로 내려도 CONTRIB-2 는 본문에서 근거 없이 남지 않는다. **캡션 의무**: F6 을 쓰는 절은 ep201 λ 급락이 CutMix 종료 시점과 겹친다는 것과, 그 원인이 LR 스케줄과 분리 확인되지 않았다는 것을 반드시 명시한다. **재평가 조건**: §9-B(LR 스케줄과의 겹침 확인)가 해소되면 F6 은 본문 승격 후보로 재검토한다 — 지금의 부록 배치는 최종 결정이 아니다 |
| F7 | Speck 실측 | 별도 머신 | **G7 — 미착수** | 본문 (있으면), 없으면 생략 | T6 과 동일 사유 |

**F2·F3·F4 캡션에 반드시 들어갈 단서 둘** (09-14 이전 세션에서 3회 반복 지적, 스펙 §8 그대로 인용):
1. 후반층(conv4, fc1)에서는 세 방법이 구분되지 않는다. 집중은 초·중반층 현상이다.
2. `dead_neuron_ratio` 는 배치 100장 기준이다. "영구히 죽은 뉴런" 으로 읽으면 안 된다.

## 층별 지니 / 상위 10% 두 세트 — 09-14 `neuron_usage.py` 세트를 쓴다

스펙 §8 이 경고한 대로 층별 지니 계수·상위 10% 점유는 두 세트가 있고 같은 층에서 값이
다르다 (예: conv2_block1 지니 — 09-10 측정 0.738/0.766/0.873 vs 09-14 `neuron_usage.py`
0.695/0.758/0.894). **F2·F3·F4 를 포함해 논문 전체에서 09-14 `neuron_usage.py` 세트를
쓴다.**

근거:
- **재생성 가능성** — `neuron_usage.py` 는 `/media/hdd1/kyccj/EIP/paper` 의 `r19c10-*` 런에서
  직접 읽어 매번 같은 수치를 재계산한다 (`gini`/`top10_share`/`dead_neuron_ratio` 열을
  `reg_neuron_detail.csv` 에서 그대로 읽음). 09-10 세트는 스펙 문서에 박제된 숫자로만
  남아 있고, 어느 런 쌍에서 나왔는지 재현 경로가 없다.
- **T1 런 집합과의 정합성** — `neuron_usage.py` 가 읽는 `r19c10-{base,l2,sm,prop}-*` 는
  T1 주 비교표가 쓰는 정본 런 계열과 같다 (`_paper/r19c10-*` 와 동일 명명 규칙). 09-10
  세트는 어느 런에서 나왔는지 스펙에 명시돼 있지 않아 T1 표와 같은 런을 쓴다는 보장이 없다.
- 09-14 세트로 F2·F3·F4 를 다시 생성해 확인함 (아래 Step 5 검증 로그의 실행 결과 참조):
  회귀 계수 0.8673(SE 0.0134, t=-10.6, n=23), 최근접 쌍 prop 193,129 / l2 193,083.

## 검증 (브리프 Step 5)

```
$ grep -oE '\bT[1-9]\b|\bF[1-7]\b' docs/paper/figures-tables.md | sort -u
F1 F2 F3 F4 F5 F6 F7 T1 T2 T3 T4 T5 T6 T7 T8 T9

$ grep -oE '\bT[1-9]\b|\bF[1-7]\b' docs/superpowers/specs/2026-09-14-eip-paper-spec.md | sort -u
F1 F2 F3 F4 F5 F6 F7 T1 T2 T3 T4 T5 T6 T7 T8 T9
```

두 출력이 같다. **T9 는 스펙 §8 이 "후보"로만 언급한 표다** — 정식 채택은 아니지만
번호가 스펙 본문에 등장하므로 이 검증에서는 포함해 부록으로 적어 둔다 (위 표 참조).

## 그림 스크립트 실행 확인 (2026-09-14)

```
$ cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py
{'base': 5, 'l2': 10, 'prop': 13, 'sm': 3}
저장: .../EIP_fig/neuron_usage_26-09-14.png
회귀: 비율 0.8673  SE 0.0134  t -10.6  n=23
쌍: prop S=193129 A=131641 | l2 S=193083 A=155669

$ /home/kyccj/anaconda3/envs/venv_1/bin/python lambda_epoch.py
저장: .../EIP_fig/lambda_epoch_26-09-14.png
λ: ep1 4.35e-07  ep100 4.85e-07  ep200 4.71e-07  ep201 2.97e-07  ep310 2.58e-07  시간평균 4.00e-07
최종 val_acc: 제안법 96.46 / 고정 3e-7 96.29 / 고정 5e-7 96.39
```

두 스크립트 모두 오류 없이 실행되어 PNG 를 갱신했다.

## F1 데이터·대표점 (Task 13, 2026-09-14)

### archive `train.log` 개수 정정

task-13-brief.md 는 `/media/hdd1/kyccj/EIP/archive` 에 `train.log` 가 15개뿐이라고
적었지만, 그 find 는 `-maxdepth 2` 였다. archive 의 실제 구조는
`archive/<family>/<run>/train.log` (예: `archive/_r19_c10_maxnorm/maxnorm_lmb_1e-07/train.log`)
로 두 단계 더 들어가야 한다. `-maxdepth 3` 으로 다시 재면 **489개**다. 브리프가 준
`_sweep_wta_rev*`·`_rho_*`·`_grp_*`·`_vm*`·`_chwise`·`_pnorm`·`_maxnorm*` 글롭은
실제로 archive 최상위에 `_sweep_wta_rev_r19_c10`·`_grp`·`_grp_smoke`·`_vmem`·`_vmlam`·
`_vmlr`·`_vmsil*`·`_pnorm`·`_r19_c10_maxnorm`·`_vgg_c10_maxnorm`·`_compare_maxnorm`
등으로 존재한다 (레포 쪽엔 `_sweep_wta_rev`·`_vmlow`·`_rho_agg` 뿐이다). `pareto_curve.py`
는 archive 를 직접 읽는다.

### 왜 ResNet19-CIFAR10 이 아니라 VGG16-CIFAR10 인가

A2(활성 뉴런 13.1%)·A3·A4·F2-F5 는 전부 ResNet19-CIFAR10 이지만, F1 은 VGG16-CIFAR10
이다. ResNet19-CIFAR10 쪽에서 실제로 재현 가능한 가중 설계는 `_r19_c10_maxnorm`
(1-maxnorm)·`_sweep_wta_rev_r19_c10`(1-softmax)·`_entropy_sweep`(entropy) 세 갈래뿐이라
"설계가 다양해도 한 곡선"이라는 A8 의 논지를 보이기엔 표본이 너무 좁다. 설계 다양성
(softmax/maxnorm/entropy/p-norm/그룹축/vmem/encourage/synops-가중 등)을 실제로 스윕한
것은 2026년 7-8월의 VGG16-CIFAR10 탐색 단계였다. 서로 다른 아키텍처를 한 곡선에 섞지
않는다 — VGG16 은 ~95%, ResNet19 는 ~97% 대가 정확도 상한이라 실제 값(비정규화) 축에서
섞으면 "한 곡선" 자체가 무의미해진다. F1 의 캡션과 본문(§4.5 "Many weighting designs,
one curve" 문단)이 이 범위 차이를 명시한다.

### 대표점 선정 규칙

설계마다, 사전 등록 선별 규칙(완주 310에폭 + S30/S1 ≥ 0.19, 스펙 §12-1)을 통과한 런
중 **검증 정확도가 가장 높은 런**을 대표점으로 쓴다. 유효 런이 하나뿐이면 그것이
대표점이고, 유효 런이 아예 없는 설계는 그림에서 빠진다. 이 규칙은 `pareto_curve.py`
의 `representative()` 함수가 기계적으로 적용하며, 설계마다 다른 판단을 넣지 않는다.

### 13종 (+baseline) 목록과 유효 런 수

`pareto_curve.py` 실행 로그(2026-09-14) 그대로:

```
baseline (규제 없음)                  acc= 95.02  spikes=    74346  (유효 1/1)
1-softmax (그룹 없음)                 acc= 95.03  spikes=    60215  (유효 4/8)
1-maxnorm (그룹 없음)                 acc= 94.65  spikes=    67481  (유효 2/4)
1-maxnorm (없음) + encourage        acc= 94.98  spikes=    68861  (유효 2/4)
1-softmax (없음) + encourage        acc= 95.05  spikes=    73812  (유효 3/4)
1-maxnorm (채널 내)                  acc= 94.72  spikes=    49963  (유효 2/3)
1-maxnorm (채널 내) + vmem           acc= 95.05  spikes=    37138  (유효 3/3)
1-maxnorm (채널 내) + vmem + silent_only  acc= 94.74  spikes=    36783  (유효 1/2)
p-norm (연속 축, p=4/8/16)           acc= 94.56  spikes=    32021  (유효 2/3)
entropy WTA                       acc= 95.11  spikes=    75635  (유효 2/4)
plain softmax (반전 없음)             acc= 95.22  spikes=    82920  (유효 4/4)
constant coefficient              acc= 95.01  spikes=    76000  (유효 2/4)
1-softmax (채널 내 그룹)               acc= 94.63  spikes=    33754  (유효 2/3)
layer_cost=synops                 acc= 94.60  spikes=    39108  (유효 1/2)

단일 곡선 적합: acc = 89.968 + 0.4506*ln(spikes),  R^2=0.518,  잔차 sd=0.160 (n=14)
```

**19종에서 13종으로 줄어든 이유** — 개요·스펙 여러 곳(§2 A8, `claim-evidence.md`,
`novelty-gate.md`)이 "19종"을 인용하지만, 실제 실행 가능한 재현 경로를 추적한 것은
이번이 처음이다. 아래 네 설계는 후보였으나 유효 런이 0개였다:

| 설계 | 위치 | 배제 사유 |
|---|---|---|
| shape_beta (β 스윕) | `archive/_beta/*` | 6개 런 전부 S30/S1<0.19 로 배제 |
| 레거시 channel-wise (row) | `archive/_channel_wise_wta/*` | 3개 런 전부 0에폭(미완주) |
| accum_loss (norm-of-sum) | `archive/_accum/*` | 4개 런 전부 S30/S1<0.19 로 배제 |
| 1-softmax(row)+flat 대조 | `archive/_grp_smoke/*` | S30/S1 은 통과하나 정확도 23.9~30.9%로 붕괴 — 그룹 구조를 고립시키는 진단용 대조군이지 시험 대상 설계가 아니어서 제외 |

**이 재검증(19종→13종, VGG16 이 논문 범위 밖)이 F1 폐기의 근거다.** 2026-09-14
사용자 결정으로 F1 은 만들지 않고, CONTRIB-1 은 A2 하나로 선다 — 등가성은 그림·수치
없는 동기 서술로만 §1 P4 에 남는다 (spec §2 A8, task-13b). 개요·`novelty-gate.md`·
`claim-evidence.md` 의 "19종" 문구는 이 결정에 맞춰 함께 수정했다.
