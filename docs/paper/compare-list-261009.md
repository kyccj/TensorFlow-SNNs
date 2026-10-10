# 비교 논문 목록 (2026-10-09)

출처: `docs/paper/refs-audit.md` 의 09-29 / 10-02 / 10-06 두 절 / 10-07 절을 **정리**했다. 수치는 그 절에서 가져왔고 이번에 다시 읽지 않았다. 추가 웹 검색 2회(스파이크 규제 + 채널 프루닝, 2025~2026 탑학회)에서는 기존 목록에 없는 새 논문이 나오지 않았다. 그래서 보충한 논문은 0편이다 (못 찾은 것이지 없다는 뜻이 아니다).

**우리 기준값.** R19 C10 T=4 baseline 96.73 (n=6, 489K spikes/sample), inter 96.74 @342K (−30%, n=3). R19 C100 T=4 80.70 (n=5, 450K). QKFormer C10 96.21 @951K → 96.13 @520K (−45%, n=2).

**공통 주의.** 아래 어느 논문도 spikes/sample 을 보고하지 않는다 (예외: All in one timestep, 추론당 총 스파이크). 보고 지표는 발화율(FR), SOPs, 이론 에너지(mJ)다. 그래서 비교는 "각 논문 안의 baseline 대비 변화"로만 한다. 모델·T 가 달라 절대값 우열 비교가 아니다. "미확인"은 본문 인용 금지.

## A. 성능 비교 표에 넣을 논문 (정확도 + 효율 지표 둘 다 보고)

비교 가능 열: ◎ 같은 모델 계열, 지표만 다름 / ○ 계열이 비슷하나 모델·T 다름 / △ 방향만 참고.

| # | 논문 (연도·학회) | 방법 한 줄 | 모델·데이터셋 | 보고 수치 (acc, 효율) | 비교 가능성 | 비고 |
|---|---|---|---|---|---|---|
| 1 | Activity Pruning, Bu/Shi/Yu (NeurIPS 2025) | AT-LIF: 출력 {0,θ}, 학습 임계값에만 규제 기울기 | R20 C10 T=8 | Table 1: 92.89→92.05 (−0.84), FR 0.236→0.053 (−77.5%), SOPs 256.69M→61.70M. 같은 표 −2.19%p 행은 FR −89.4% | ○ 목표 같음, 모델·T 다름 | 코드 공개 (github.com/putshua/Activity-Pruning-SNN). n 미기재. **R19 T=4 재현 1순위** |
| 2 | 〃 | 〃 | SEW-R18 ImageNet-100 T=4 | 83.44→82.40 (−1.04), FR −72.5%, SOPs 1276M→464M | △ ImageNet-100 | SDTv3 ImageNet 과 모델 다름 |
| 3 | 〃 | 〃 | VGGSNN CIFAR10-DVS T=10 | 82.70→82.20 (−0.50), FR 0.080→0.043 (−46.3%), SOPs −41.2% | ◎ VGGSNN DVS | 우리 DVS 수치 나오면 같은 T·증강으로 맞춤 |
| 4 | UniSparse, Shi/Bu/Yu (ICML 2026) | 에너지 가중 스파이크 벌점, 기울기는 BN γ,β(=임계값)로, 가중치 프루닝 병행 | R19 C10 T=4 | Table 1: 94.71±0.11 @39.47M SOPs → 94.07 @18.58M → 91.56 @6.11M | ◎ R19 C10 T=4 | **밀집 baseline 행이 표에 없음**, 스파이크 단독 효과 분리 안 됨. n 미확인. 코드 공개, `cifar10_res19.yaml` 있음 |
| 5 | 〃 | 〃 | VGGSNN DVS T=10 | 83.22±0.53 @39.26M → 79.12±0.83 @5.65M | ◎ | 위와 같은 한계 |
| 6 | 〃 | 〃 | ImageNet R18 T=4 / Spikformer-8-384 | R18 61.89 @275M → 59.66 @109M; Spikformer 72.21 @2577M → 68.29 @1139M | △ | SDTv3 파인튜닝과는 모델 다름 |
| 7 | Spik4lite, She 외 (ICML 2026) | 채널 발화율 라우터 + Gumbel 게이트, 에너지 벌점, 채널 물리 제거 | Spikformer-4-384 C10 T=4 | Table 1: 95.24→94.15, SOPs 822.6M→321.5M (−60.9%), Jetson 5.34→3.08 J | ○ 트랜스포머 계열 | **정확도 전 행 −0.63~−3.81%p.** Jetson Orin Nano 실측 에너지. n 없음 |
| 8 | 〃 | 〃 | Spikingformer-4-384 C10 / C100 | C10 95.07→94.43 (SOPs −36.7%); C100 79.07→77.07 | ○ | QKFormer 는 지원 안 함 |
| 9 | MI-TRQR, Xue 외 (NeurIPS 2025) | 시간 스텝 간 중복 스파이크를 확률 마스크로 제거, 무파라미터 플러그인 | PLIF-Net C10 T=4 | Table 7·8: 95.32→95.83±0.10, 에너지 1.32→0.35 mJ, **FR 16.39%→5.36% (−67.3%)** | ○ 방향 같음, baseline 약함 | 기준선 PSN. 코드 공개 (dfxue/MI-TRQR). 분류 표 ± 의 n 미확인. 에너지는 이론값 |
| 10 | 〃 | 〃 | R19 C100 T=4 | Table 7: 76.14→77.70±0.10, 1.78→1.24 mJ (♠ 저자 재구현) | ◎ R19 C100 T=4 | 우리 baseline 80.70 보다 4.6%p 낮음 |
| 11 | 〃 | 〃 | STMixer-4-384 C10 / C100 T=4 | C10 96.01→96.64±0.12, 0.95→0.51 mJ; C100 81.87→83.06, 1.08→0.77 mJ | ○ 트랜스포머류 (QKFormer 아님) | |
| 12 | 〃 | 〃 | VGG CIFAR10-DVS T=4 | 82.3→83.9±0.05, 0.72→0.45 mJ (−37.5%); Table 4·6 FR 14.47%→8.35% | ○ DVS | T=4 (우리 T 와 확인 필요) |
| 13 | SCA, Li 외 (ICML 2024) | 채널 활동 기반 구조적 프루닝 + 재성장 | VGG16 C10 / C100, 5Conv+1FC DVS | C10 (기준 91.14) +0.32@파라미터 55.9%, +0.53@28.4%, −0.88@9.3%; DVS (72.80) +0.90@21.7% | △ 기준 정확도 낮음 | SynOps 는 그림뿐. 채널 단위 구조적 주장이 inter 와 겹침 |
| 14 | UPR, Shi 외 (ICLR 2024 Spotlight) | 비구조적 뉴런 프루닝 | CifarNet C10 T=8; DVS | 92.63 @38.3M SOPs (SpikeNM/UniSparse 표 인용); DVS 82.4→81.9, SOPs 47.8M | △ | 원 논문 표는 이번에 안 읽음(2차 인용). 가장 직접적 경쟁자 |
| 15 | Suetake 외 (TMLR 2024, 게재 제목 "Synaptic Interaction Penalty") | 에너지에 비례하는 시냅스 벌점 | ResNet-18 C10 | arXiv 판 Table 10: 91.32±0.21→91.40±0.29 @E 58%; 89.46 @E 19% | △ 구조가 가장 닮음 | **게재본 수치 미확인.** 정확도 SOTA 급 아님. 우리가 직접 재현한 baseline |
| 16 | BPSR, Yan 외 (Front. Neurosci. 2022) | 제곱 L2 스파이크 희소화 | (우리가 직접 실험) | 외부 수치는 이번 목록에 없음 | ◎ 직접 비교 실험 | 서지 저자·쪽수 미확인 |
| 17 | Deng 외 (IEEE TNNLS) | 평균 스파이크율에 λR | 10층 CNN C10 | Table IV: λ=0 89.53 → 0.001 89.51 (FR 0.11 불변) → 0.01 87.62 (FR 0.08) | △ plain 선례 | 감소하면 −1.91%p 이상. 게재처는 2차 출처 |
| 18 | Allred 외 (arXiv 2020, 게재 미확인) | 총 스파이크 수에 σ 스케줄 벌점 | LeNet C10 | 검증 오차 무증가 조건 스파이크 −67~−69% | △ 저정확도 모델 (64.97%) | 인용만 |
| 19 | CH-SNN (ICLR 2026) | 연결 희소 학습 | 3FC C10 | 스파이크 6.3e7→1.9e7 (−70%), 80.67→82.84 | △ 비교 불가 | 스파이크 수를 표로 보고하는 드문 논문. 3FC, 가중치 희소화 |
| 20 | Uncert (arXiv 2605, 게재 미확인) | QKFormer 토큰 가지치기 | QKFormer C10 T=4 | 96.31→96.30, 블록 SOP −14.86%, 전체 1.457→1.406 mJ (−3.5%) | ◎ **QKFormer** | 감소 폭이 우리 −45% 보다 훨씬 작음 |
| 21 | TP-Spikformer (ICLR 2026) | 재학습 없는 토큰 가지치기 | Spikformer-4-384 C10/C100 T=4; QKFormer ImageNet | C10 95.19→95.12 (토큰 ×0.20); ImageNet QK-10-768 85.56→82.53, 32.12→25.71 mJ | ○ | QKFormer 는 ImageNet 만 |
| 22 | STATA (ICML 2024) | 토큰 솎기 | Spikingformer C100 | 79.98 @1.29 mJ → 79.86 @0.71 mJ | ○ | |
| 23 | STAS (arXiv, 게재 미확인) | 블록×시간 정지 | Spikformer-4-384 C10/C100 T=4 | 94.8/77.3 @0.74/0.89 mJ → 95.2/77.9 @0.40/0.50 mJ | ○ | |
| 24 | SparseSpikformer (ICASSP 2024) | 토큰+가중치 프루닝 | Spikformer C10 | 94.95 @3.74G → 95.01 @3.04G (−19%) | ○ | |
| 25 | SpikeNM (arXiv 2511, 게재 미확인) | N:M 가중치 프루닝 | C10 6Conv+2FC T=8; DVS VGGSNN T=10 | C10 2:8 92.92 @196M SOPs; DVS 83.90(+1.50) @289M | △ | **SOPs 는 UPR 쪽이 훨씬 낮다.** 스파이크 감소 목적 아님 |
| 26 | DS-ATGO (AAAI 2026) | 임계값 적응으로 정확도 향상 | R19 T=2 C10 | 96.91±0.12, AC 443M→691M (0.43→0.65 mJ) | ◎ R19 | **반례 용도**: 정확도 올리면 스파이크도 는다. T=2 |
| 27 | MTT (ICLR 2025) | T 유연 학습 | R19 C10/C100 T=4 | 96.75±0.04 / 81.51±0.04 (3회) | ◎ R19 T=4 | 효율 지표 없음. **정확도 기준선 확인용** (우리 96.73/80.70 과 같은 선) |
| 28 | SEENN (NeurIPS 2023) | 샘플별 조기 종료 | R19 C10/C100 | C10 96.07@평균 T 1.09 … 96.44@1.34 (T=4 기준 96.46) | ○ T 축 | 스파이크 수 아님. "T 를 줄이면 되지 않나" 반론용 |
| 29 | Logit-KD (ICML 2025) | 로짓 증류로 소 T | R19 C10/C100 T=4/2 | 96.97 / 82.47 (T=4), 96.65 (T=2) | ◎ R19 | 스파이크 수 미보고. 교사 ANN 필요 |
| 30 | QKFormer (NeurIPS 2024) | 원 모델 | QKFormer-4-384 C10/C100 T=4 | 96.18 / 81.15 | ◎ 기준선 | 효율 지표 CIFAR 에 없음. 우리 96.21 과 일치 = 재현 근거 |

**표 A 행 수: 30** (그중 우리와 같은 모델 계열에 놓을 수 있는 ◎ 는 3, 4, 5, 10, 16, 20, 26, 27, 29, 30).

**표에서 빠진 것 (이유):** SDTv3 ImageNet 파인튜닝과 같은 모델·설정으로 규제를 보고한 논문은 찾지 못했다 (E-SpikeFormer/SDT-V3 는 ImageNet 정확도 중심, 스파이크 규제 표 없음, 게재 미확인). 행 2, 6 은 ImageNet 계열 참고용이다. 정확도만 있고 효율 지표가 없는 TET/RMP-Loss/MPBN 등은 기준선 표로만 쓰고 위에는 넣지 않았다.

### C-1. 표 A 에서 우리 주장에 위험한 논문

1. **MI-TRQR (행 9~12).** 정확도 +0.5~1.7 에 FR −67%, 에너지 −37~−73% 를 플러그인 하나로 주장한다. "스파이크를 줄이면서 정확도가 오른다" 가 우리 핵심과 같은 방향이고 감소율이 우리 R19 −30%@±0.00 보다 크다. 방어: 기준선이 PSN 95.32 (우리 96.73 보다 낮음), 에너지는 이론값, 규제항이 아니라 마스크 모듈, 분류 ± 의 n 불명. 같은 R19 T=4 에서 재현하지 않으면 우열을 못 쓴다.
2. **Activity Pruning (행 1~3).** 감소율(FR −77%)이 우리 최대보다 크고, "일반 스파이크 규제는 task 와 충돌해 정확도를 희생한다" 는 논지를 정면으로 쓴다. 우리 주장(규제항 하나로 무손실)의 직접 반례 후보다. 방어: baseline 92.89 (R20 T=8), 정확도 −0.84%p, 뉴런 모델 변경. 같은 설정 재현이 필요하다.
3. **UniSparse (행 4~6).** 같은 연구실의 SOTA 주장이고, 인트로에서 스파이크 규제를 "무차별 억제" 라고 비판한다. 우리 경쟁 가중이 이 비판에 답해야 한다. 표에 밀집 baseline 이 없어 손실 폭을 우리가 계산할 수 없다는 점이 약점이면서, 리뷰어가 "R19 C10 T=4 에서 이미 94% @6M SOPs" 를 들이밀 수 있다는 위험이다.

(Spik4lite 는 실측 에너지를 낸다는 점에서 SynOps 프록시 한계를 지적받을 때 위험하지만, 전 행 정확도 −0.6~−3.8%p 라 "무손실" 주장 자체는 위협하지 않는다.)

## B. Related Work 에서 인용·대조할 논문

관계: 선행 = 우리와 같은 일을 먼저 했음 / 대조 = 다른 방식으로 같은 목표 / 보완 = 결합 가능.

| 분류 | 논문 | 관계 | 한 줄 차이점 |
|---|---|---|---|
| 스파이크 규제 | Suetake, TMLR 2024 (Synaptic Interaction Penalty) | 선행 | 에너지 비례 벌점, 균등 가중·수동 λ. 우리는 채널 간 경쟁 가중 + 자동 λ |
| 〃 | BPSR, Yan, Front. Neurosci. 2022 | 선행 (직접 비교 대상) | 제곱 L2 균등 벌점. 경쟁 가중 없음 |
| 〃 | Deng 외, IEEE TNNLS (ADMM + Activity Reg.) | 선행 | 스파이크율 λR 선형. 같은 논문 표에서 감소하면 −1.91%p |
| 〃 | Sorbaro 2020; Pellegrini 2021; Sakemi Sci. Rep. 2023; Neil 2016; Allred 2020 | 선행 | 모두 균등 가중·수동 λ, 소형 모델 위주 |
| 〃 | Stanojevic, Nat. Commun. 2024 | 대조 | TTFS L1, VGG16 C10 93.69→93.27. 부호화 다름 |
| 〃 | Kurtz, ICML 2020 (Hoyer) | 선행 (ANN) | 활성에 스케일 불변 노름. SNN·채널 경쟁 없음 |
| 〃 | UniSparse, ICML 2026 | 대조 + 비판자 | 에너지 가중 L1을 BN 아핀(임계값)에 직접 갱신. 가중치 프루닝 혼합 |
| 활동 프루닝 | Activity Pruning, NeurIPS 2025 | 대조 | 뉴런 모델(AT-LIF)을 바꿔 해결. 우리는 뉴런 그대로 |
| 〃 | UPR, ICLR 2024 Spotlight | 선행 | 비구조적 뉴런 프루닝. 사후 단계·별도 기준 |
| 〃 | MI-TRQR, NeurIPS 2025 | 대조 | 마스크 모듈로 시간 중복 제거. 규제항 아님 |
| 구조적·채널 | SCA, ICML 2024 | 선행 | 활동 기반 채널 프루닝 + 재성장. 우리는 규제만으로 채널이 죽음 (inter) |
| 〃 | Spik4lite, ICML 2026 | 대조 | 라우터·게이트 + 별도 손실, 물리 제거, Jetson 실측. 정확도 하락 |
| 〃 | Kundu Spike-Thrift, WACV 2021; DPAP (arXiv); SPEAR (ICLR 2026 투고만 확인) | 선행 | 규제가 아니라 프루닝 단계. SPEAR 채택 미확인 |
| 〃 | SASL (arXiv 2020, 게재 미확인) | 선행 (ANN) | 약한 필터일수록 큰 벌점(5등급). BN γ 대상. 연속 가중·활성 대상 아님 |
| 〃 | QP-SNN, ICLR 2025; LTH, ECCV 2022 | 보완 | 가중치 프루닝은 스파이크를 줄이지 않는다 (LTH Fig 10) |
| 측면 억제·WTA | FBS, ICLR 2019 | 선행 (ANN) | 채널 k-WTA 하드 게이트 + lasso. 우리는 손실만 |
| 〃 | SpiLiFormer, ICCV 2025 | 대조 | 어텐션 구조에 측면 억제. 효율 지표 초록에 없음 |
| 〃 | DeepEISNN, ICLR 2026 | 대조 | E-I 회로로 정규화 대체. 스파이크 감소 수치 없음 |
| 〃 | LISNN, IJCAI 2020; WTA Spiking Transformer (arXiv 2604.11321, 게재 미확인); EIHL, ICLR 2024 | 대조 | 구조 변경. EIHL 희소도는 가중치 연결 |
| λ 자동 조정 | DAP (arXiv 2608, 게재 미확인); SymbolNet (arXiv, 게재 미확인) | 선행 (비 SNN) | λ ∝ L_task/L_reg. SNN·활성 규제 아님. "SNN 스파이크 규제에서 처음" 으로 한정 |
| 에너지·SynOps 기준 | Shen 외, CVPR 2024 (Bit Budget) | 대조 | SynOps 대신 비트 예산. 수치 표 미확인 |
| 〃 | Reconsidering the Energy Efficiency of SNNs (arXiv 2409.08290) | 대조 | 에너지 프록시 비판 |
| 〃 | The Sparsity Ceiling (arXiv 2607.26648) | 대조 | 희소성 이득은 과제 속성 (지각 5%, 순환 LM 50%, 스파이킹 트랜스포머 2%) |
| 〃 | All in one timestep (arXiv 2510.24637, 게재 미확인) | 보완 | 총 스파이크 수가 에너지 핵심 지표라고 주장 → 우리 spikes/sample 정당화 인용 |
| T 축 | SEENN NeurIPS 2023; DT-SNN (학회 미확정); Dynamic Confidence ICCV 2023 | 보완 | 평균 T 감소. 발화율 축과 곱해질 수 있음 |
| 임계·막전위 | RMP-Loss ICCV 2023; IM-Loss NeurIPS 2022; TET ICLR 2022 | 대조 | 막전위/출력 분포 손실. 스파이크 수 직접 벌점 아님 |

### C-2. 표 B 에서 우리 주장에 위험한 논문

1. **SASL + FBS (ANN).** "경쟁(순위) 가중은 새롭다" 라고 쓰면 반례다. SASL 은 약한 필터에 큰 벌점을, FBS 는 채널 k-WTA 로 쓰지 않는 채널 제거를 이미 했다. 새로움은 "SNN 스파이크 규제에서 처음" 으로만 한정해야 한다.
2. **Suetake (TMLR 2024).** "규제항 하나" 라는 구조가 가장 닮아 새로움 선점 후보다. 게재본 수치를 못 읽었으므로 읽기 전에는 차별점(경쟁 가중, 자동 λ, 정확도 유지)을 수치로 방어할 수 없다.
3. **The Sparsity Ceiling (arXiv 2607.26648) / Reconsidering the Energy Efficiency (arXiv 2409.08290).** 스파이크 감소가 곧 에너지 이득이라는 전제를 부정한다. 스파이킹 트랜스포머 2% 라는 수치는 QKFormer·SDTv3 실험에 직접 걸린다. SynOps 프록시 한계를 우리가 먼저 명시해야 한다.

## 미확인 (본문 인용 금지)

Suetake TMLR 게재본 수치; UniSparse 밀집 baseline·n; MI-TRQR 분류 표 ± 의 n 및 PSN 서지; Spik4lite ± 유무; Shen CVPR 2024 수치; BPSR 저자·쪽수; Deng TNNLS 저널 페이지; SPEAR 채택; DAP·SymbolNet·SASL·Uncert·SpikeNM·STAS·WTA Spiking Transformer 게재 여부; UPR 원 논문 표 (이번엔 2차 인용). 코드 공개를 이번에 확인한 것은 Activity Pruning, UPR, UniSparse, Spik4lite, MI-TRQR, DS-ATGO, MTT, EfficientST(refs-audit 기재 기준).
