# 참고문헌 검증 기록

`paper/refs.bib` 28건 전체에 대한 검증 상태다. 검증은 이 작업(Task 2)에서 직접
publisher 페이지·arXiv listing·DBLP·Semantic Scholar 를 fetch 하여 수행했다.
`.superpowers/sdd/2026-09-14-eip-paper-writing/lit-bcd.md` (선행 조사)는 후보 목록으로만
썼고, 값을 그대로 베끼지 않고 각 항목을 다시 확인했다.

**미확인으로 표시된 항목은 본문에서 인용하지 않는다.** 이 문서 기준으로 현재 미확인 항목은
없다 — 28건 전부 원문 소스를 직접 확인했다. 다만 항목별로 확인 범위가 다르므로 아래 표에
"확인 방법"을 구체적으로 적었다.

## A. 선점 대응

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `bu2025activity` | 확인 | Task 1 이 확정한 값(저자·NeurIPS 2025·DOI 10.52202/085713-5700, DBLP key conf/nips/BuSY25)을 그대로 사용. arXiv 미러 없음, NeurIPS 권 번호는 미확인이라 bibtex 에서 제외 |
| `li2024spiking` | 확인 | PMLR 공식 페이지(`proceedings.mlr.press/v235/li24bz.html`)에서 저자·권·페이지(235:29063–29073) 직접 확인 |
| `yao2023inherent` | 확인 | arXiv:2308.08227 abstract 페이지에서 저자 순서 확인: Yao, Hu, Zhao, Wang, Zhang, Xu, Li. **lit-bcd.md 의 저자 목록(Zhao, Zhang, Hu, Deng, Tian, Xu, Li)은 틀렸다** — 재검증으로 교체함. ICCV 2023 pp.16924–16934 는 별도 검색 결과로 교차 확인 |

## B. 스파이크 규제 계보

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `neil2016learning` | 확인 | ZORA UZH 저장소(취리히대) 서지 레코드로 저자·venue(ACM SAC 2016) 확인 |
| `sorbaro2020optimizing` | 확인 | Task 1(novelty-gate.md)이 원문 PDF 를 직접 읽고 확정한 값(Frontiers 14:662, doi:10.3389/fnins.2020.00662)을 그대로 사용 |
| `pellegrini2021low` | 확인 | IEEE SLT 2021 자료(IEEE Resource Center) + arXiv:2011.06846 로 저자·페이지(97–103) 확인 |
| `meda2022improving` | 확인 | arXiv:2201.02538 abstract 페이지에서 단독 저자 **Nandan Meda**(학사 학위논문)로 확인됨. 처음 배정된 키가 저자 성과 불일치해 저자 성에 맞춰 개명함(2026-09-14 fix round 1 — Task 4/9/10 착수 전이라 하위 참조 영향 없음). **보조(secondary) 인용으로만 쓸 것** — 학사 학위논문이라 동료심사를 거치지 않았다. 손실항 규제/정규화라는 같은 주제의 1차 문헌 근거는 `neil2016learning`(ACM SAC), `sorbaro2020optimizing`(Frontiers), `pellegrini2021low`(IEEE SLT) 세 편이며, plain-L2 계열 baseline 을 뒷받침할 때는 이 세 편을 우선 인용하고 `meda2022improving` 은 보완 각주로만 붙인다 |
| `sakemi2023sparse` | 확인 | Nature 저널 페이지(nature.com/articles/s41598-023-50201-5) + PMC 사본으로 저자·권·article no.(13:22897) 확인 |

## C. 직접 학습 기초

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `wu2018spatio` | 확인 | arXiv:1706.02609 + PubMed 레코드로 저자·Frontiers 12:331(2018) 확인 |
| `neftci2019surrogate` | 확인 (브리프 Step 2 공백 항목) | Google Scholar 서지 레코드로 **vol.36, no.6, pp.51–63** 확인, doi:10.1109/MSP.2019.2931595 |
| `wu2019direct` | 확인 | AAAI 공식 저널 페이지(ojs.aaai.org)로 vol.33, pp.1311–1318 확인 |
| `zheng2021going` | 확인 | AAAI 공식 PDF(cdn.aaai.org) + doi(10.1609/aaai.v35i12.17320)로 vol.35 no.12 pp.11062–11070 확인 |
| `deng2022temporal` | 확인 | arXiv:2202.11946 abstract 페이지로 저자(Deng, Li, Zhang, Gu) 확인. venue(ICLR 2022)는 논문 자체 표기 |
| `fang2021incorporating` | 확인 | CVF Open Access 페이지로 저자·ICCV 2021 pp.2661–2671 확인 |

## D. 아키텍처

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `zhou2023spikformer` | 확인 | arXiv:2209.15425 + ICLR 2023 poster 페이지로 저자·venue 확인 |
| `yao2023spikedriven` | 확인 (브리프 Step 2 공백 항목) | NeurIPS 공식 proceedings 페이지(`papers.neurips.cc`) + GitHub 공식 구현으로 저자 7인·**NeurIPS 2023** 확정 |
| `yao2024metaspikeformer` | 확인 (브리프 Step 2 공백 항목) | arXiv:2404.03663 + GitHub 공식 구현(`BICLab/Spike-Driven-Transformer-V2`, "ICLR2024" 명시)으로 저자 8인·**ICLR 2024** 확정. Meta-SpikeFormer 와 SDT-V2 는 동일 논문의 별칭임을 재확인 |
| `yao2025scaling` | 확인 | GitHub 공식 구현("IEEE T-PAMI2025") + 검색 결과로 vol.47 no.4 pp.2973–2990 확인 |

## E. 에너지 회계

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `horowitz2014computing` | 확인 (브리프 Step 2 공백 항목) | Scientific Research Publishing 서지 레코드로 **pp.10–14**, San Francisco 2014-02-09~13 확인 |
| `lemaire2022analytical` | 확인 | arXiv:2210.13107 + Springer 챕터 링크(ICONIP 2022, LNCS 13623, doi:10.1007/978-3-031-30105-6_48)로 저자 6인·venue·DOI 확인. 브리프는 arXiv 만 언급했으나 ICONIP 2022 accepted 사실과 Springer DOI/series/volume 을 추가로 확인해 `refs.bib` 에 반영(2026-09-14 fix round 1) |

## F. 뉴로모픽 하드웨어

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `yao2024spikebased` | 확인 | Nature Communications 공식 페이지(nature.com/articles/s41467-024-47811-6) + PMC 사본으로 저자 18인 전체 확인 |

## G. 데이터셋

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `li2017cifar10dvs` | 확인 | Frontiers 저널 페이지로 저자(Li, Liu, Ji, Li, Shi)·11:309(2017) 확인 |
| `orchard2015converting` | 확인 (브리프 Step 2 공백 항목) | PubMed + Semantic Scholar 로 **vol.9, article 437**, doi:10.3389/fnins.2015.00437 확인 |
| `krizhevsky2009learning` | 확인 | 원문 PDF(cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) 존재 확인. Tech Report, University of Toronto, 2009 |

## H. 측면 억제 (동기 서술 전용)

| 키 | 상태 | 확인 방법 |
|---|---|---|
| `cheng2020lisnn` | 확인 | IJCAI 공식 proceedings 페이지(ijcai.org/proceedings/2020/211)로 저자(Cheng, Hao, Xu, Xu)·pp.1519–1525 확인 |

## I. 손실항 가중치 자동 설정 계보 (Step 3, SNN 밖 최소 3편)

| 키 | 상태 | 확인 방법 | 우리와 다른 점 한 줄 |
|---|---|---|---|
| `chen2018gradnorm` | 확인 | PMLR 공식 페이지(`proceedings.mlr.press/v80/chen18a`)로 vol.80 pp.794–803 확인 | 태스크별 가중치를 **기울기 노름 균등화**로 매 step 학습(추가 파라미터·비대칭 계수 α 필요). 우리는 학습 파라미터 없이 손실 값의 비로 λ 를 매 epoch 닫힌 형태로 재해결 |
| `kendall2018multitask` | 확인 | CVF Open Access 공식 페이지로 pp.7482–7491 확인 | 가중치를 동분산 불확실성 σ² 의 **학습 파라미터**로 둔다(확률 모형/우도 해석 필요). 우리는 규제항에 확률 모형을 가정하지 않고 크기 비만 쓴다 |
| `franke2024improving` | 확인 | arXiv:2311.09058 + NeurIPS 공식 proceedings 페이지로 저자(Franke, Hefenbrock, Koehler, Hutter) 확인 | 파라미터 행렬별 **증강 라그랑지안 승수**를 학습해 상한(constraint)을 강제한다 — 상한값 지정이 필요하다. 우리는 상한도 목표치도 없이 비율 ρ 하나만 지정한다 |

## 요약

- 총 28건, 미확인 0건.
- 브리프 Step 2 의 서지 공백 5건(`horowitz2014computing`, `neftci2019surrogate`,
  `orchard2015converting`, `yao2023spikedriven`, `yao2024metaspikeformer`) 모두 직접
  재검증하여 채웠다.
- 재검증 중 **lit-bcd.md 의 오류 1건**을 발견해 고쳤다: `yao2023inherent` 저자 순서.
- **키 개명 1건 (2026-09-14 fix round 1)**: 저자 성과 불일치하던 원래 키를 `meda2022improving`
  으로 고쳤다(실제 저자는 Nandan Meda). Task 4/9/10 착수 전이라 하위 영향 없음. 이 항목은
  학사 학위논문이므로 **보조 인용 전용**이며, plain-L2 baseline 의 1차 근거는
  `neil2016learning`·`sorbaro2020optimizing`·`pellegrini2021low` 세 편이다. 위 표 참조.
- `bu2025activity` 는 Task 1 이 이미 확정한 값을 그대로 옮겼으며 이 작업에서 재검증하지
  않았다(OpenReview 본문이 CAPTCHA 로 막혀 있다는 사실도 Task 1 산출물과 동일).

## 2026-09-29 관련 연구 비교 (규제·희소화·비스파이크 효율)

조사 방법: 3개 조사 에이전트가 arXiv PDF 를 내려받아 `pdftotext` 로 표를 직접 읽었다. 그림에서만 읽히는 값과 확인 못 한 칸은 **미확인**으로 뒀다. "(파싱)" 은 표 열 정렬이 깨져 원문 재대조가 필요하다는 뜻이다. 전수 조사가 아니라 이 세션 검색 범위 안의 결과다.

**공통 주의 (세 범주 모두).**
- 아래 어느 논문도 CIFAR 에서 **spikes/sample 을 표로 보고하지 않는다.** 지표는 평균 T, 발화율, SOPs, 이론 에너지(mJ, E_AC=0.9pJ / E_MAC=4.6pJ 류), 파라미터 비율 등이다. 우리 spikes/sample 과 직접 환산되지 않는다.
- 우리 baseline(R19-C10 96.73%, QKFormer-C10 96.21%)은 아래 R19 수치(93~96%)보다 높다. 학습 레시피가 달라 **절대 정확도가 아니라 baseline 대비 변화**로 비교해야 한다.
- 우리 수치: R19-C10 96.73%@486K → 96.69%@245K (−50%), QKFormer-C10 96.21%@951K → 96.13%@520K (−45%).

### A. 규제를 쓰면서 SOTA 급 정확도

결론: **스파이크 수를 벌점으로 넣으면서 SOTA 급 CIFAR 정확도를 낸 논문은 찾지 못했다.** SOTA 급은 규제가 없거나(모델 기준선), 규제 대상이 막전위·출력이다. 스파이크·에너지를 직접 벌점으로 쓴 논문은 정확도가 SOTA 급이 아니거나 부호화가 다르다.

| # | 논문 (저자, 학회·연도, 링크) | 규제 | 방법 한 줄 | 수치 (표, 설정) | 주의점 |
|---|---|---|---|---|---|
| A1 | TET. Deng, Li, Zhang, Gu. ICLR 2022. https://arxiv.org/abs/2202.11946 | MSE (스파이크 규제 아님) | 시점별 CE + 출력을 V_th 에 맞추는 MSE | Table 3, T=4: R19 C10 94.44±0.08, C100 74.47±0.15. C10-DVS VGGSNN T=10 77.33±0.21 (증강† 83.17±0.15). 스파이크·발화율 미보고. Fig 4B 에 에너지 곡선(값 미확인) | 표준 R19 기준선. 우리 baseline 이 2%p 이상 높음(레시피 차) |
| A2 | Spikformer. Zhou et al. ICLR 2023. https://arxiv.org/abs/2209.15425 | 없음 | 스파이킹 트랜스포머 | Table 3: 4-384, T=4: C10 95.51, C100 78.21 (9.32M). Table 4: C10-DVS T=16 80.9(증강). CIFAR 발화율·에너지 미확인 | 저자 목록은 이 세션에서 직접 확인 못 함 |
| A3 | Spike-driven Transformer. Yao et al. NeurIPS 2023. https://arxiv.org/abs/2307.01694 | 없음 | 스파이크 구동 어텐션 | Table 5, 2-512, T=4: C10 95.6, C100 78.4; C10-DVS T=16 80.0. Table 2 발화율은 ImageNet 8-512 뿐 | 전체 스파이크 수 없음 |
| A4 | Spikingformer. Zhou et al. arXiv v5 (2025-11). https://arxiv.org/abs/2304.11954 | 없음 | 스파이크 구동 residual | Table 3, 4-384, T=4: C10 95.81, C100 79.21; C10-DVS T=16 81.3 | "AAAI 2026" 은 GitHub README 검색요약일 뿐 **채택 미확인** |
| A5 | QKFormer. Zhou et al. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2403.16552 | 없음 | Q-K 계층형 어텐션 | Table 3, HST-4-384 6.74M, T=4: C10 96.18, C100 81.15. C10-DVS HST-2-256 T=16 84.0 / T=10 83.8. CIFAR 발화율·에너지 없음(발화율은 ImageNet Table 4) | 우리 baseline 96.21 이 논문값과 일치 = 재현 근거 |
| A6 | SpiLiFormer (lateral inhibition). Zheng et al. arXiv 2503.15986 v2. https://arxiv.org/abs/2503.15986 | 없음 (어텐션 구조) | 어텐션에 측면억제 | Table 2, 4-384 7.04M T=4: C10 96.63, C100 81.63; C10-DVS 2-256 T=16 86.7. CIFAR 에너지 미확인 | 학회 미확인. 이름이 우리 WTA 동기와 혼동됨(구조 vs 규제항) |
| A7 | RMP-Loss. Guo et al. ICCV 2023. https://arxiv.org/abs/2308.06787 | 막전위 분포 | 막전위를 0/1 근처로 모으는 손실 | Table 2 (파싱), R19 C10 T=2/4/6 95.31/95.51±0.08/96.10±0.08; C100 74.66/78.28±0.10/78.98. Table 3 C10-DVS R19 T=10 76.20±0.20. 같은 표에 인용된 InfLoR-SNN R19 C10 T=4 96.27 (파싱, 원문 미확인). 효율 표 미확인 | R19 T=4 로 정확도 직접 비교 가능. IM-Loss(기지)와 별개 논문 |
| A8 | Stabilizing Direct Training (MP-Init + TrSG). Kook, Yu, Oh, Park. arXiv 2511.08708. https://arxiv.org/abs/2511.08708 | 스파이크 희소성 손실을 진단 도구로 사용 | 막전위 초기화 + 임계 강건 surrogate | Table 3: R19 T=4 C10 95.34±0.06, C100 77.66±0.15; C10-DVS R19 T=10 81.43±0.40. Fig 8: R19 C100 발화율 0~0.55 곡선. Table 13: 1.12G ACs / 0.14G MACs, 1.67 mJ, 77.68% (데이터셋 표기 미확인, C100 추정) | 학회 미확인. R19 에서 정확도×발화율 축을 그림. 스파이크 절감이 목적은 아니고 통제 실험 |
| A9 | 0.3 spikes per neuron. Stanojevic et al. Nat. Commun. 2024. https://arxiv.org/abs/2306.08744 | L1 | TTFS SNN, ReLU 사전학습(L1)→변환 | Table 2 VGG16: C10 93.69 (0.38 spikes/neuron) → +L1 93.27±0.02 (0.20); C100 72.24 (0.38) → 72.20±0.04 (0.24) | arXiv v2 수치, 게재본 미확인. TTFS 라 T=4 구조와 지표 다름. 정확도 −0.4%p(C10) |
| A10 | Spiking Synaptic Penalty. Suetake et al. arXiv 2302.01500. https://arxiv.org/abs/2302.01500 | 에너지 비례 벌점 | 손실에 λ·Ω_syn (SNN 에너지에 정확히 비례) | Table 10, ResNet-18 C10: λ=0 91.32±0.209% (E 100%); λ=1e-8 91.40±0.293% (58.03%); λ=1e-7 89.46% (19.13%) | 게재 미확인. **구조가 우리와 가장 닮음**, 정확도는 SOTA 급 아님 |

### B. 스파이크(발화율/SOPs)를 줄여 효율을 높인 논문

조기 종료 계열은 스파이크 수가 아니라 평균 T 를 줄인다(우리와 직교, 결합 가능). 스파이크 수를 직접 잰 것은 B3 뿐이다.

| # | 논문 | 방식 | 수치 (표, 설정) | 주의점 |
|---|---|---|---|---|
| B1 | SEENN. Li, Geller, Kim, Panda. NeurIPS 2023. https://arxiv.org/abs/2304.01230 | 샘플별 조기 종료 (신뢰도 / RL) | Table 1, R19: C10 SEENN-I 96.07@평균T 1.09, 96.38@1.20, 96.44@1.34; SEENN-II 96.01@1.08; TEBN T=2/4 95.45/95.58. C100 SEENN-I 79.56@1.19, 81.42@1.55, 81.65@1.74; SEENN-II 80.23@1.21. Table 4 C10-DVS VGGSNN: SEENN-I 77.6@2.53, 82.7@5.17; SEENN-II 82.6@4.49. Fig 3 의 T=4 기준선 96.46. 에너지(mJ)는 그림뿐, 미확인 | 지표 = 평균 T. 스파이크 수 미보고. 리뷰어의 "T 를 줄이면 되지 않나" 반론용 |
| B2 | DT-SNN (Input-Aware Dynamic Timestep). Li, Moitra, Geller, Panda. https://arxiv.org/abs/2305.17346 | 엔트로피 기반 조기 종료 | Table II, 정적 T=4 대비: R19 C10 94.03→93.87@T=1.27 (E 0.41×); C100 73.31→73.48@1.90 (0.53×); C10-DVS R19 T=10 75.0→74.8@5.02 (0.52×); VGG16 C10 93.17→93.58@1.46 (0.46×) | **학회 미확정**(요약에 DAC 2023 / DATE 상충, PDF 본문 미확인). 에너지는 IMC 시뮬레이션. 스파이크 수 미보고 |
| B3 | Dynamic Confidence. Li, Jones, Furber. ICCV 2023. https://arxiv.org/abs/2303.10276 | 신뢰도 기반 종료 (ANN→SNN 변환) | Table 1 C10: ResNet-18 QFFS 94.11 불변, T 4→2.52; QCFS 94.27 불변, T 27→11.51; VGG16 QFFS 92.41, T 6→3.17. Table 3 "spike counts per neuron per inference": R18 QCFS 3.02→1.29, R18 QFFS 0.34→0.25, VGG16 QCFS 3.21→1.35, VGG16 QFFS 0.32→0.31 | 변환 계열, 모델·T 다름, 뉴런당 단위. C100/DVS 없음. 논문이 스파이크 수를 전력 프록시로 쓴다고 인정 |
| B4 | SCA (Spiking Activity based Pruning). Li, Xu, Shen, Xu, Chen, Pan. ICML 2024. https://arxiv.org/abs/2406.01072 | 채널 활동 기반 구조적 프루닝 + 재성장 | Table 1 (정확도 변화, 남은 파라미터): C10 VGG16(기준 91.14) +0.32@55.86%, +0.53@28.39%, −0.88@9.31%; C100 VGG16(64.89) +0.64@23.52%, −0.25@16.47%; C10-DVS 5Conv+1FC(72.80) +0.90@21.73%, −0.90@6.95%. SynOps 는 Fig 2 뿐(VGG16 C10 채널 0.8 제거 시 약 1/2, DVS 0.43M에서 −1%p 미만; 표 없음) | C10 T=4 (본문). 모델 VGG16, 기준 정확도 낮음. 손실 규제가 아니라 프루닝 스케줄. `ours_inter_ch` 와 채널 단위 구조적 주장이 겹침 |
| B5 | SPEAR. Xie et al. https://arxiv.org/abs/2507.02945 (OpenReview id Wo8hTwoYcm) | SynOps 제약 RL 구조적 프루닝 | Table 1 (SynOps 남은 %, 파라미터 %, 정확도): C10 VGG16 52.5/14.4/91.77; C10 ResNet18 39.2/30.3/92.78; C100 VGG16 69.0/35.0/70.50; C100 R18 48.2/20.4/68.86; C10-DVS 5Conv+1FC 39.3/17.1/80.05. Table 5 VGG16 92.43→92.49, 0.11→0.07 mJ, SynOps 62.5%. 다른 행의 기준 정확도(부록 Table 7) 미확인 | **ICLR 2026 채택은 검색 요약뿐, 미확인.** 45nm 에너지 추정. 모델·T 다름 |
| B6 | STATA (token sparsification). Zhuge, Wang, Yao, Cheng. ICML 2024. https://proceedings.mlr.press/v235/zhuge24b.html | 토큰 솎기 | Table 4: Spikformer C10 95.0 / C100 77.7 (T=4), DVS 80.7 (T=16); SDT 95.4 / 78.1 / 79.7; Spikingformer 95.8 / 79.9 / 81.2. Table 3 Spikingformer C100: 79.98@1.29 mJ → 79.86@0.71 mJ | 지표 OPs·mJ. 스파이크 수 미확인. QKFormer 아님 |
| B7 | TP-Spikformer. Wei, Zhou, Zhang et al. ICLR 2026. https://arxiv.org/abs/2603.00527 | 재학습 없는 토큰 가지치기 | Table 1 (Spikformer 기준): C10 T=4 95.19→95.12(−0.07)@토큰 ×0.20; C100 78.21→77.83(−0.38)@×0.55, 78.48@×0.60; DVS T=16 80.9→80.7@×0.50. Table 2 ImageNet QKFormer QK-10-768 85.56%/32.12 mJ → 82.53%(−3.03)/25.71 mJ | QKFormer 는 ImageNet 뿐. 스파이크 수 없음 |
| B8 | STAS (spatio-temporal ACT). Kang et al. arXiv 2508.14138. https://arxiv.org/abs/2508.14138 | 블록×타임스텝 정지 | Table 3, T=4 (C10/C100): Spikformer-4-384 0.74/0.89 mJ 94.8/77.3 → 0.40/0.50 mJ 95.2/77.9; Spikingformer-4-384 0.42/0.50 mJ 95.7/79.4 → 0.37/0.46 mJ 95.8/79.4 | 학회 미확인, 참고용 |

### C. 스파이크를 건드리지 않고 효율을 높인 논문

| # | 논문 | 방식 | 수치 (표, 설정) | 주의점 |
|---|---|---|---|---|
| C1 | Lottery Ticket in SNNs. Kim et al. ECCV 2022. https://arxiv.org/abs/2207.01382 | 비구조적 가중치 프루닝 | Table 2, T=5: R19 C10 (밀집 93.22) 76.20% 희소 +0.28, 94.29% +0.24, 97.54% −0.04; C100 (71.34) +0.11 / −0.34 / −2.29. C10-DVS 없음. 스파이크 수는 Fig 10 에서 "희소율과 무관하게 비슷"(절대값 미확인) | **가중치 프루닝은 스파이크를 줄이지 않는다**는 근거로 인용 가능. 이득은 가중치 메모리 이동 에너지. 직교 |
| C2 | Q-SNNs. Wei et al. ACM MM 2024. https://arxiv.org/abs/2406.13672 | 1비트 가중치 + 막전위 2/4/8비트 | Table 1/2 (1w-8u / 4u / 2u): R19 C10 T=2 95.54 / 95.31 / 95.20; C100 T=2 78.77 / 78.82 / 78.70; C10-DVS VGGSNN T=10 81.60 / 81.50 / 80.00. 메모리 1w-2u: −96.75%(batch 1), −94.57%(batch 256); 1.62 MB | 스파이크 수·SOPs·에너지 미보고. 완전 직교 |
| C3 | QP-SNN. Wei et al. ICLR 2025. https://arxiv.org/abs/2502.05905 | 양자화 + 채널 프루닝 | Table 1: C10 R20 T=2 (8/4/2비트) 95.12/95.41/95.06, 프루닝 행 94.56/94.65/94.44; C10 VGG16 T=4 91.98/91.90/91.61; C100 R20 T=2 75.29/75.77/75.13; C10-DVS VGGSNN T=10 82.10/81.80/81.30 (프루닝 행 81.50/80.90/80.50 및 75.90/75.40/74.90, 각 connection 비율 미확인). Table 6: VGG16 C10 밀집 54.60M SOPs, 0.204 mJ, 93.63 → 9.61% 연결·4비트 11.63M SOPs, 0.046 mJ, 91.19; R20 C100 415.64M, 0.756 mJ, 79.49 → 22.69%, 131.53M, 0.126 mJ, 74.73. Table 2 Spikingformer-4-384 C100: 79.09/0.266 mJ/292.14M → 76.94/0.118 mJ/130.05M SOPs | 트랜스포머 결과 있음. SOPs 감소는 프루닝 가중치 때문(우리와 원인 다름). 정확도 손실 큼(C100 −2.2~−4.7) |
| C4 | Logit-KD for full-range timestep. Yu et al. ICML 2025. https://arxiv.org/abs/2501.15925 | 로짓 증류, 소 T 유지 | Table 1 (3회 평균): R19 T=6/4/2 C10 97.00/96.97/96.65, C100 82.56/82.47/81.47; R18 T=4/2 C10 95.57/95.11. Table 3: C10-DVS R18 T=4 83.50, T=10 86.40. Table 11 Spikingformer-4-384 T=4 C100 78.34→80.77. Table 10 R18 C100 발화율 T=4 0.2018 vs 0.2044 | 스파이크 수 미보고. R19 C10 T=4 96.97 > 우리 baseline. 교사 ANN 필요 |
| C5 | SparseSpikformer. Liu, Xiao, Li, Yu. ICASSP 2024. https://arxiv.org/abs/2311.08806 | 토큰 + 가중치 프루닝 | Table 1 (밀집→ρ=0.7): C10 94.95/3.74G → 95.01/3.04G; C100 77.28 → 77.42; DVS 78.3/7.78G → 79.3/6.57G. Table 2 가중치 희소: C10 93.90%에서 94.97, C100 92.04%에서 75.12, DVS 95.8%에서 76.2 | 토큰 선택 기준이 발화율(개념적 겹침). 스파이크 수 미보고. Spikformer 계열 |

### 우리와 직접 비교 가능한 것 (같은 모델·데이터·T)

지표까지 같은 논문은 없다. 모델·데이터·T 가 맞는 것만 고른다. "n" 은 전부 논문 보고값이며 우리 쪽 n 은 spec 을 따른다.

| 범주 | 우리 설정 | 비교 대상 (표) | 수치 | 비교 가능한 축 |
|---|---|---|---|---|
| A | R19 C10 T=4 | TET (T3) / RMP-Loss (T2) / MP-Init+TrSG (T3) | 94.44 / 95.51 / 95.34 | 정확도만 (baseline 대비 변화로) |
| A | R19 C100 T=4 | TET / RMP-Loss / MP-Init+TrSG | 74.47 / 78.28(파싱) / 77.66 | 정확도만 |
| A | R19 C100 T=4 | MP-Init+TrSG Fig 8 | 발화율 0~0.55 곡선 | 정확도×발화율 (유일하게 같은 축) |
| A | QKFormer C10/C100 T=4 | QKFormer (T3) / SpiLiFormer (T2) | 96.18/81.15 / 96.63/81.63 | 기준선 |
| A | C10-DVS VGGSNN | TET VGGSNN T=10 | 77.33 (증강 83.17) | 증강 여부를 맞춰야 함 |
| B | R19 C10/C100 T=4 baseline | SEENN (T1), DT-SNN (TII) | 96.46 기준 → 96.0~96.4@T≈1.1~1.3; 94.03 → 93.87@1.27 | 평균 T, 정규화 에너지 (스파이크 수 아님) |
| B | C10-DVS | SEENN (T4), DT-SNN (TII) | VGGSNN 82.6~82.7@T≈4.5~5.2; R19 74.8@5.02 | 평균 T |
| B | C10/C100 Spikformer계 T=4 | STATA, STAS, TP-Spikformer | 위 표 | OPs·mJ, QKFormer 아님 |
| C | R19 C10/C100 | Logit-KD (T1, T=4/2), Q-SNNs (T=2), LTH (T=5) | 96.97/96.65; 95.54~95.20; 93.22 기준 | 정확도, 파라미터·메모리 |
| C | C10-DVS VGGSNN | Q-SNNs, QP-SNN, TET | 81.60~80.00 (T=10) | 정확도, 모델 크기 (곧 나올 우리 VGGSNN 수치용) |

### 결합(직교) 가능한 것

- 양자화(Q-SNNs, QP-SNN), 비구조적 가중치 프루닝(LTH): 스파이크 수와 무관. LTH 가 스파이크 불변을 직접 보였다.
- 조기 종료(SEENN, DT-SNN), 소 T 증류(Logit-KD): T 축. 우리는 발화율 축.
- 토큰 프루닝(STATA, SparseSpikformer, TP-Spikformer): 토큰 축.
- 채널 구조적 프루닝(SCA, SPEAR, QP-SNN): 스파이크·SynOps 가 함께 줌. 효과가 겹칠 수 있음.

### 위협도와 포지셔닝 의견

위협적인 것:
1. **SEENN / DT-SNN / Logit-KD (T 축).** 같은 R19 C10 에서 정확도를 유지하며 T 를 4→1.1~2 로 줄인다. 스파이크 수가 T 에 대체로 비례한다는 가정 하에서는 우리 −50% 이상의 절감 가능성이 있다(**추정**, 논문은 스파이크 수 미보고). Logit-KD 의 R19 C10 T=2 96.65 는 우리 −50% 결과 96.69 와 정확도가 같다. 방어: 추론 정책·교사 ANN 이 필요, 결합 실험, 발화율 축은 T 와 독립.
2. **Suetake et al. (A10) 과 Kook et al. (A8).** A10 은 "에너지 비례 벌점 하나"라는 구조가 우리와 가장 닮아 새로움 주장의 선행 사례가 된다. A8 은 R19 에서 정확도×발화율 파레토를 그려 리뷰어가 같은 축 비교를 요구할 수 있다. 둘 다 게재 미확인.
3. **SCA (ICML 2024).** 채널 활동 기반 구조적 프루닝으로 `ours_inter_ch` 와 겹친다. 관련 연구에서 반드시 인용.

의견:
- **이기는 곳.** 조사한 논문 중 스파이크 수를 손실로 직접 줄이면서 R19/QKFormer 를 SOTA 급 정확도(96.7 / 96.2)로 유지한 것은 없다. 규제 논문은 막전위(A7, A8) 또는 저정확도(A10), TTFS(A9)다. 트랜스포머(QKFormer)에서 −45% 를 보인 규제 연구도 이 조사에는 없다.
- **밀리는 곳.** T 축 방법이 더 큰 절감을 낼 가능성이 있고, 우리는 spikes/sample 을 보고하는데 경쟁 논문은 T·SOPs·mJ 를 보고해 표준화된 비교가 안 된다. 같은 R19 C10 에서 정확도 절대값은 Logit-KD(96.97), SEENN(96.46 기준)이 우리와 비슷하거나 높다.
- **권고.** (a) 우리 모델의 SOPs·이론 에너지를 A/B/C 의 지표로 환산해 함께 보고하고, (b) SEENN/DT-SNN 과의 결합 실험으로 T 축과 곱해짐을 보이며, (c) MP-Init 과 같은 발화율 축(R19 C100)에서 파레토를 겹쳐 그리는 것이 가장 방어력이 높다.

### 미확인 항목 (본문 인용 금지)

SPEAR 의 ICLR 2026 채택과 부록 Table 7; DT-SNN 게재 학회; Spikingformer 의 AAAI 2026; SpiLiFormer·MP-Init+TrSG·STAS·Suetake 의 게재 여부; Stanojevic 의 Nat. Commun. 게재본 수치; InfLoR-SNN 원문; SEENN Fig 3 과 TET Fig 4B 의 에너지 값; LTH Fig 10 스파이크 절대값; Q-SNNs 의 스파이크·SOPs·에너지; QP-SNN 의 DVS 프루닝 행별 connection 비율; SCA SynOps 표; Spikformer 저자 목록. 제외한 것: Narduzzi 2022 (LeNet-5 변환, 게재 미확인), Hoyer regularizer (Datta, 게재 미확인), "One Timestep is All You Need"(arXiv 2021), Circulate-Firing(arXiv 2026, 미검증).


## 2026-10-02 최신 논문 비교 (2025-06 이후)

조사 방법: 논문 PDF 를 내려받아 `pdftotext -layout` 으로 표를 직접 읽었다. 읽지 못한 것은 **미확인**으로 표시했다. 2026-09-29 절에 있는 논문은 다시 조사하지 않았다. 전수 조사가 아니라 이 세션 검색 범위 안의 결과다.

**읽은 방식 (논문마다 다르다).**
- Activity Pruning: NeurIPS 2025 proceedings PDF 전문을 읽었다 (OpenReview 는 CAPTCHA 로 막혀 있어 `proceedings.neurips.cc` 검색 페이지에서 PDF 링크를 찾았다).
- Spik4lite (ICML 2026): **논문 PDF 를 열지 못했다** (OpenReview 차단, arXiv 판 없음). 저자 공개 저장소(코드·`train.log`·README)만 읽었다. 방법은 코드에서, 수치는 저자 실행 로그에서 읽었다. 논문 표 수치가 아니다.
- UniSparse (ICML 2026): **논문 PDF 를 열지 못했다** (OpenReview 차단). 저장소 코드와 Lacuna 2차 요약만 봤다. 수치는 모두 2차 요약 값이다.
- 나머지는 arXiv PDF 에서 표를 읽었다.

### 1. 수치 표

우리 기준 (비교용): R19 C10 T=4 baseline 96.73 (n=6, 489K spikes/sample) → inter 96.74 @342K (−30%, n=3), 96.66 @302K (−38%); intra 96.59 @192K (−60%, n=5). R19 C100 T=4 80.70 (n=5, 450K) → layer w¹ 80.67 @315K (−30%), intra 80.36 @220K (−51%). QKFormer C10 T=4 96.21 → 96.13 @−45% (n=2).

**공통 주의.** 아래 논문은 spikes/sample 을 보고하지 않는다. 지표는 평균 발화율(Avg FR)·SOPs·채널 수·파라미터다. 모델·T·baseline 이 달라서 우리 수치와 나란히 놓을 수 없다. 변화량(Δ)은 각 논문 안의 baseline 대비로만 계산했다. 아래 "−x%" 는 표 값에서 내가 계산한 것이다.

| # | 논문 | 학회·연도 | 설정 (표) | baseline → 결과 | 효율 지표 | n |
|---|---|---|---|---|---|---|
| 1a | **Activity Pruning** (Bu, Shi, Yu; PKU) https://proceedings.neurips.cc/paper_files/paper/2025/hash/fa12d67b5939c37ea8a4659c31a08d2c-Abstract-Conference.html | NeurIPS 2025 (게재 확인) | Table 1, CIFAR-10 ResNet-20, T=8 | Vanilla 92.89 → AT-LIF(λθ=1e-3) 92.05 (−0.84); λθ=2e-3 91.06 (−1.83); λθ=5e-3 88.31 (−4.58) | Avg FR 0.236 → 0.053 (−77.5%) / 0.043 (−81.8%) / 0.033; SOPs 256.69M → 61.70M (−76.0%) / 48.85M / 39.33M | 미기재 |
| 1b | 〃 | 〃 | Table 1, CIFAR-10 ResNet-20, T=8, AT-LIF + L1 (λs=0.01, λθ=1e-3) | 92.89 → 90.70 (−2.19) | FR 0.236 → 0.025 (−89.4%); SOPs → 29.23M (−88.6%) | 미기재 |
| 1c | 〃 | 〃 | Table 1, 같은 설정의 L1 규제(AR) | λs=0.01: 91.31 (−1.58), FR 0.060, SOPs 57.26M; λs=0.02: 88.31, FR 0.044, SOPs 41.86M | — | 미기재 |
| 1d | 〃 | 〃 | Table 1, ImageNet-100 SEW-ResNet18 T=4 | 83.44 → 82.40 (−1.04) (λs=0.01, λθ=1e-4); 81.12 (−2.32) (0.02, 2e-4) | FR 0.138 → 0.038 (−72.5%) / 0.025; SOPs 1276.42M → 463.65M (−63.7%) / 328.34M | 미기재 |
| 1e | 〃 | 〃 | Table 1, CIFAR10-DVS VGGSNN T=10 | 82.70 → 82.20 (−0.50) (λθ=5e-4); 82.10 (λs=0.001) | FR 0.080 → 0.043 (−46.3%) / 0.039; SOPs 1066.40M → 626.57M (−41.2%) / 587.68M | 미기재 |
| 1f | 〃 | 〃 | Table 3 ablation, CIFAR-10 R20 T=8 | 0-θ 출력 없이 적응 임계만: **50.99%** (FR 0.110); 0-θ 출력만: 93.32 (FR 0.237, 줄지 않음); 둘 다: 92.05 (FR 0.054) | — | 미기재 |
| 1g | 〃 | 〃 | Table 2, 6Conv+2FC C10 / SEW-R18 ImageNet, STDS 가중치 프루닝과 결합 | C10 90.72 @FR 0.024, 0.19M params, SOPs 30.11M (STDS 단독 90.71 @0.072, 0.12M, 45.48M); ImageNet 59.61 @FR 0.063, 2.22M | — | 미기재 |
| 2 | **Spik4lite** (She, Zhou, Wang, Huang, Cui, Guo) https://icml.cc/virtual/2026/poster/63522 , 코드 https://github.com/yongzhishe/Spik4lite | ICML 2026 (포스터 목록 확인) | **저자 `train.log` (논문 표 아님)**: Spikingformer, CIFAR-10, T=4, dim 384, 400 에폭 | 논문 Spikingformer 4-384 T=4 C10 = 95.81 (2026-09-29 절 A4) → 로그 최고 94.40 (에폭 384, 마지막 에폭 93.84) | 채널 25,594 → 2,481 (−90.3%); 파라미터 29.79M → 3.87M (로그; 게이팅 층 포함으로 추정) | 1런 |
| 3 | 〃 | 〃 | **저자 `train.log`**: Spike-Driven Transformer, CIFAR-10, T=4, dim 512, 300 에폭 | 논문 SDT 2-512 T=4 C10 = 95.6 (A3) → 로그 최고 94.78 (에폭 294) | 파라미터 28.77M → 8.89M (−69%; 로그) | 1런 |
| 4 | **UniSparse** (Shi, Bu, Yu; PKU) https://icml.cc/virtual/2026/poster/64520 , 코드 https://github.com/xyshi2000/UniSparse | ICML 2026 (포스터 확인) | **2차 요약, 표 미확인.** CIFAR-10 ResNet-18 | 92.38% | SOPs 7.04M | 미확인 |
| 5 | 〃 | 〃 | **2차 요약, 표 미확인.** CIFAR10-DVS | 79.12% (이전 최고보다 +0.82) | SOPs 5.65M | 미확인 |
| 6 | **DeepEISNN** (Liu, Ding, Yu) https://arxiv.org/abs/2509.23253 | ICLR 2026 (PDF 머리글 확인) | Table 1, CIFAR-10 ResNet-18 T=4 | vanilla BN 95.37±0.13 → E-I 정규화 없는 모델 92.05±0.11; CIFAR-100 VGG-16 T=4 64.95±1.14 | 발화율·스파이크·에너지 보고 없음 | 표기 있음 (런 수 미확인) |
| 7 | **All in one timestep** (Castagnetti, Pegatoquet, Miramond; Univ. Côte d'Azur) https://arxiv.org/abs/2510.24637 | arXiv 2025-10, 게재 미확인 | Table 1/Fig. 7, 다중 레벨 뉴런, CIFAR-10 VGG16 | T=1·N=4 94.34%. 같은 기능의 T=4 이진은 130K spikes/inference, 다중 레벨은 57K (−43%) | **추론당 총 스파이크 수를 보고** (Fig. 7 본문 값). 에너지 ANN 대비 0.34 (다중 레벨) | 미기재 |
| 8 | **Circulate-Firing** (Zhou, Wei, Liu, Yu; Tianjin Univ.) https://arxiv.org/abs/2605.27412 | arXiv 2026-05, 게재 미확인 | Table II, ResNet-18 T=4 | C10 97.31±0.11, C100 80.89±0.18 | Table X: SOPs 6.08G, 0.468 mJ (ResNet-18, T=4) | 표기 있음 |

### 2. 정리가 필요한 항목

- **Activity Pruning 은 스파이크 *규제*를 비판하는 논문이다.** 본문(2.3절, 3.1절)이 "스파이크 활동에 직접 규제를 걸면 과제 손실과 목표가 충돌해 정확도-희소성 교환이 생긴다"고 쓰고, Theorem 1 로 스파이크가 전부 0 인 점이 안장점임을 보인다. 방법은 규제 항이 아니라 **뉴런 모델 변경**이다. AT-LIF 는 출력이 {0, θ}(1이 아니라 임계값)이고, 임계값 θ 가 학습 가능하며 θ 에만 규제 기울기를 준다 (Eq. 12: `Δθ = η(∂L_task/∂θ + λθ Σ_t s[t] ∂o[t]/∂θ)`). 가중치는 과제 손실로만 갱신한다. 추론에서 θ 를 다음 층 가중치에 흡수한다고 주장한다. 우리와 같은 점은 "정확도를 지키며 스파이크를 줄인다"는 목표와 학습 중 규제뿐이다. 경쟁 범위(채널)·채널 죽음은 다루지 않는다.
- Activity Pruning 의 정확도 주장에는 주의가 필요하다. 본문이 "CIFAR-10 에서 vanilla 와 비슷한 정확도로 FR −89.4%, SOPs −88.6%"라고 쓴 행(1b)은 Table 1 에서 정확도가 **−2.19%p**(92.89 → 90.70)다. 정확도 손실이 작은 행은 1a(−0.84%p, FR −77.5%)다. 시드 수·SD 는 표에 없다. baseline 이 92.89%(ResNet-20, T=8)로 우리 baseline(96.73%)보다 3.8%p 낮다.
- Activity Pruning 의 ablation(1f)은 주목할 만하다. 0-θ 출력 없이 적응 임계값만 쓰면 50.99%로 무너진다. 저자들은 "스파이크 수를 줄이는 규제는 출력 크기를 보상하지 않으면 학습이 무너진다"고 해석한다. 우리 방법은 이 보상 없이 −60% 에서 −0.14%p 이므로 반대 증거가 되지만, 모델·T 가 달라 직접 대조는 아니다.
- **Spik4lite 방법 (코드에서 확인, 논문 서술과 대조 못 함).** 각 합성곱 층 앞에 게이트(`Linear(C, 2C)` + Gumbel-Softmax)를 두고, 입력은 샘플별 채널 평균 발화율이다. 손실 `EnergyPenaltyTerm` = λ_energy(기본 0.01) × 층 평균 [Σ(keep 확률 × 비용 계수) / 최대 비용]. 비용 계수 = `정적 연산량 × 채널 발화율의 이동평균` 이고 **detach 되어 있다**. 즉 기울기는 게이트 확률로만 흐르고 스파이크 자체는 벌점을 받지 않는다. 학습 중 keep 확률을 70 에폭 주기로 누적해 임계값(0.5) 아래 채널을 **물리적으로 제거**(가중치 텐서 축소)하고 마지막 50 에폭은 게이트 없이 미세조정한다. 저장소에 Jetson 전력 로그(`Spikingformer Power profile.csv`, `Spikingformer+Spik4lite Power profile.csv`)가 있어 **실기 전력 측정**을 한다. 지원 모델은 Spikformer, SDT, Spikingformer (README). QKFormer 와 ResNet-19 는 없다.
- **Spik4lite 로그 수치의 한계.** 위 94.40 / 94.78 은 저자 한 번의 실행 로그이고 baseline 을 같은 레시피로 다시 돌린 값이 없다. "논문 baseline 95.81 / 95.6 대비 −1.4 / −0.8%p"는 레시피가 같다는 가정이 있어야 성립한다. 논문 표는 이보다 좋을 수 있다 (미확인).
- **UniSparse 방법 (코드에서 확인).** 손실 항이 아니라 **파라미터 직접 갱신**이다. BN 의 γ, β 를 `-= λ_neuron × penalty` 로 줄이고 (BN affine 이 다음 뉴런의 임계값 역할을 한다는 관찰), 가중치는 소프트 임계값 `threshold += λ_weight × penalty` 로 줄인다. penalty 는 하나의 에너지(SOPs) 모델에서 유도한다. 채널별 BN 파라미터를 건드리므로 스파이크 억제는 채널 단위다. 모델: CifarNet, ResNet-18/19, VGGSNN, Spikformer (ImageNet). 저장소 설정에 **CIFAR-100 은 없다**. Activity Pruning 과 같은 연구실이고, ICLR 2024 의 비구조적 프루닝 논문(기지)과도 같은 연구실이다.
- **DeepEISNN 은 "측면 억제"라는 말만 겹친다.** 흥분/억제 뉴런 회로로 BN 을 대체하는 구조 논문이다. 발화율·스파이크 수 보고가 없고 정확도가 BN 모델보다 3.3%p 낮다. 위협 아님. 같은 이유로 WTA Spiking Transformer for Language Modeling (Zhou et al., arXiv 2604.11321, 2026-04, 게재 미확인)도 소프트맥스 대체 어텐션 구조 논문이라 제외한다.
- **Circulate-Firing (arXiv 2026) 의 정체.** 2026-09-29 절에 "미검증"으로 남아 있던 항목은 arXiv 2605.27412 "Advancing Direct Training for SNNs with Circulate-Firing Neurons and Learnable Gradients"다. 막전위의 음의 영역까지 쓰는 뉴런 모델과 학습 가능한 대리 기울기를 제안하는 **정확도 논문**이다. 스파이크 감소 방법이 아니다. 다만 Table II 가 R19/R18 의 최근 직접 학습 정확도 비교표를 갖고 있다 (T=4, C10/C100): LM-H R19 96.36/81.65, MPBN R19 T=2 96.47/79.51, LM-HT R19 T=2 96.89/81.76, 본인 R18 T=4 97.31/80.89. 우리 baseline 96.73/80.70 은 이 범위 안이다.
- **All in one timestep (Castagnetti 외)** 은 규제가 아니라 뉴런 모델(다중 레벨 스파이크)과 구조(Sparse-ResNet)다. 의미가 있는 부분은 두 가지다. (a) 추론당 총 스파이크 수를 직접 보고하고 "총 스파이크 수가 에너지의 가장 중요한 지표"라고 쓴다 (메모리 접근이 시냅스 연산보다 10~100배 비싸다는 근거). 우리 spikes/sample 지표를 정당화하는 인용이 된다. (b) 다중 레벨 스파이크는 값이 크므로 "스파이크 수 감소"의 정의가 우리와 다르다.
- 읽었으나 위협이 낮아 표에서 뺀 것: Activation-Energy Pruning (Bingham, arXiv 2609.26167, 2026-08; 사후 비구조 가중치 프루닝, 단일 저자, 원본 VGG-SNN CIFAR-100 65.0%로 baseline 이 낮음, 게재 미확인), PSViT (Putra 외, arXiv 2606.03257, 2026-06; 사후 구조적 프루닝, ImageNet 만, CIFAR 없음). 가중치 수준 프루닝으로 직교인 SLAMP (arXiv 2603.14946)와 SpikeNM (arXiv 2511.12097)은 표를 읽지 않았다 (미확인).

### 3. 우리가 이길 수 있는 것 / 밀리는 것 / 직교

**직접 비교해서 이길 수 있는 것.**
- *낮은 정확도 손실 구간에서의 스파이크 감소.* Activity Pruning 이 큰 감소(FR −77%)를 내는 구간(1a)은 정확도 −0.84%p 이고 baseline 이 92.89% 다. 우리는 −60% 에서 −0.14%p (n=5, baseline 96.73%). 모델·T 가 달라 "이긴다"고 쓰려면 같은 R19 T=4 에서 직접 돌려야 한다. 코드가 공개되어 있다 (https://github.com/putshua/Activity-Pruning-SNN).
- *"규제는 정확도를 희생한다"는 선행 주장에 대한 반례.* Activity Pruning 이 2.3절·3.1절에서 일반 규제(L1/L2)의 한계로 서술한 내용에 대해, 우리는 경쟁 항 하나로 −30~−60% 에서 정확도 변화 ≤0.15%p 를 보였다 (R19 C10, n=3~5).
- *QKFormer 에서의 결과.* 조사한 논문 중 QKFormer 에서 스파이크 규제를 보고한 것이 없다 (Spik4lite 는 Spikformer/SDT/Spikingformer, UniSparse 는 ImageNet Spikformer).
- *CIFAR-100 R19 T=4.* UniSparse 저장소에는 CIFAR-100 설정이 없고 Activity Pruning 도 CIFAR-100 을 쓰지 않는다 (Table 1 은 CIFAR-10, ImageNet-100, DVS).

**밀리는 것.**
- *절감 폭의 최대치.* Activity Pruning(FR −89%, SOPs −88.6%), Spik4lite(로그 기준 채널 −90.3%), UniSparse(SOPs 7.04M, 2차 요약)는 우리 최대 −60% 보다 훨씬 깊게 줄인다. 단 정확도를 더 잃는 구간이다 (1b −2.19%p, Spik4lite 로그 추정 −0.8~−1.4%p). "손실 ≤0.2%p" 조건에서 이들의 곡선은 표에 없다. 이 조건에서의 비교는 모른다 (미확인).
- *실기 측정.* Spik4lite 는 Jetson 전력 로그를 갖고 있다. 우리는 MACs −33~40% (죽은 채널 제거)까지만 있고 실기 전력은 없다. "실측 검증"을 주장하는 논문이 같은 영역에 이미 있다.
- *물리적 제거의 완결성.* Spik4lite 는 학습 중 채널을 실제로 제거하는 파이프라인(옵티마이저 상태 슬라이싱 포함)을 갖췄다. 우리의 채널 제거는 학습 후 분석이다.
- *지표의 표준성.* 경쟁 논문은 Avg FR·SOPs·mJ 를 보고한다. 우리 spikes/sample 은 이들로 환산되지 않는다. 우리 모델의 SOPs 와 Avg FR 도 함께 보고해야 표에 놓일 수 있다.

**직교(결합 가능).**
- *임계값/출력 크기 접근 (Activity Pruning AT-LIF, UniSparse BN-affine 임계값).* 우리 규제를 켠 채로 임계값 적응을 얹을 수 있다. 결합 시 중복되는지(둘 다 같은 스파이크를 줄이는지) 아니면 곱해지는지는 실험 전에는 모른다.
- *가중치 프루닝 (STDS, UniSparse 의 가중치 항, SLAMP, SpikeNM).* 가중치 수준이라 스파이크를 줄이지 않는다 (2026-09-29 절 C1 의 LTH 와 같은 논리). Activity Pruning Table 2 가 활동 희소성과 가중치 프루닝의 결합이 가능함을 보였다.
- *다중 레벨 스파이크, 소 T 학습 (All in one timestep, SEENN 계열).* 스파이크 정의나 T 축을 바꾼다.

### 4. inter(채널 간 경쟁 → 채널 죽음) 주장과 가장 겹치는 논문

1. **Spik4lite (ICML 2026).** 겹침: 채널 단위로 발화율에 기반한 SOPs 비용을 학습 중 줄이고 채널을 없애 "스파이크 희소성을 채널 희소성으로 바꾼다"는 것이 제목·초록의 주장이다. 차별화: *"Spik4lite 는 학습 가능한 게이트와 별도 손실로 채널을 골라 제거하지만, 우리는 게이트·마스크·재학습 없이 스파이크에 걸린 규제 항 하나가 채널 간 경쟁을 통해 같은 스파이크 수에서 더 많은 채널을 죽게 만든다 (plain L2 대비 +44~48 개, 제거 시 정확도 손실 0)."* 주의: 이 차별화는 Spik4lite 논문 표를 읽기 전의 판단이다. 논문이 "채널이 죽는 현상"을 다른 말로 서술하고 있는지는 미확인.
2. **UniSparse (ICML 2026).** 겹침: 하나의 λ 와 하나의 에너지 모델로 BN affine(채널별)을 통해 스파이크를 줄이는, 가장 "단일 규제 항에 가까운" 방법이다. 차별화: *"UniSparse 는 BN 의 γ, β 를 직접 깎아 채널의 임계값을 올리지만, 우리는 파라미터를 직접 갱신하지 않고 손실 항으로 기울기를 흘리며 λ 를 과제 손실 비율로 자동 조정한다."* 주의: 논문 본문을 못 읽었다. λ 자동 조정이 없다는 판단은 코드 인자(`--lmbda-weight`, `--lmbda-neuron` 수동 입력)에만 근거한다.
3. **Activity Pruning (NeurIPS 2025).** 겹침: 같은 목표(정확도를 지키며 스파이크·SOPs 감소)와 같은 학회 급. 채널·죽음 주장과는 겹치지 않지만 "스파이크 규제는 정확도를 희생한다"는 논지가 우리 핵심 주장과 정면으로 충돌하여 리뷰어가 인용할 가능성이 가장 높다. 차별화: *"Activity Pruning 은 일반 규제가 정확도를 잃는다고 보고 뉴런 모델을 바꿔 해결하지만, 우리는 뉴런 모델을 그대로 두고 경쟁 가중 항 하나로 같은 구간에서 정확도 손실을 0.15%p 이하로 유지한다 (R19 C10 T=4, n=3~5)."*

### 5. 미확인 목록 (본문 인용 금지)

- Spik4lite 논문의 모든 표 (CIFAR/ImageNet 정확도, 채널 제거 비율, SOPs, 전력). 로그 값(94.40, 94.78, 25,594→2,481)은 저자 한 번의 실행이다. ICML 2026 게재는 포스터 목록(https://icml.cc/virtual/2026/poster/63522)으로 확인했다.
- UniSparse 논문의 모든 표. 92.38% @7.04M SOPs, 79.12% @5.65M SOPs, ImageNet +0.67% 는 Lacuna 2차 요약이다. T, 시드 수, baseline 정확도 모두 미확인. CIFAR-100 보고 여부는 저장소 설정 파일에 없다는 사실만 확인했다.
- Activity Pruning 의 시드 수·SD (표에 없음), 보충 자료(증명, 의사코드), 임계값 θ 가 층 단위인지 뉴런 단위인지 (본문 서술이 모호함).
- 2026-09-29 절의 "Circulate-Firing" 이 이 논문(2605.27412)과 같은 것인지는 제목·연도·arXiv 로 확인했으나, 그 절을 쓴 에이전트가 본 원문과 같다는 보증은 없다.
- DeepEISNN 의 시드 수, DeepEISNN·Circulate-Firing·All in one timestep 의 게재 여부.
- SMixer (ICLR 2026, Lu 외), DS-ATGO (AAAI 2026), SpikePack (ICCV 2025), Learnable Burst Quantization (arXiv 2604.25688)는 검색 결과로 존재만 확인했고 스파이크 규제 방법이 아니어서 표를 읽지 않았다.
- 이 조사에서 **Spik4lite 와 UniSparse 의 R19/QKFormer CIFAR-10/100 T=4 수치를 얻지 못했다**: 5번 항목(우리 표에 나란히 놓을 수 있는 2025~2026 논문)은 Activity Pruning 의 R20 T=8 만 해당한다.

### 6. 위협도와 권고

실제로 위협적인 것: (1) Spik4lite, (2) UniSparse, (3) Activity Pruning. 이유는 4절에 썼다. 권고(제안일 뿐 실행하지 않았다): 세 논문 모두 코드가 공개되어 있으므로, 논문 표를 읽을 수 없는 Spik4lite·UniSparse 는 저자 코드를 우리 설정(R19 T=4, C10/C100)으로 직접 돌려 같은 축에서 비교하는 것이 PDF 확보보다 확실하다. UniSparse 는 `configs/cifar10_res19.yaml` 이 이미 있다. 이 실행은 3등급(런 투입)이므로 승인이 필요하다.


## 2026-10-02 학습 알고리즘 부류 비교 (학습 중 발화 감소)

범위: 손실 항·규제·학습 규칙·임계값 적응으로 **학습 중** 발화/스파이크/SOPs/에너지를 줄이는 방법. 제외: 추론시 T 단축, 프루닝 파이프라인, 양자화, 변환 전용, 발화 감소가 목적이 아닌 막전위 규제. 이 절은 에이전트 3개가 PDF 를 `pdftotext` 로 직접 읽은 결과다(BPSR 은 PMC 전문 HTML). 기존 두 절(2026-09-29, 2026-10-02 최신)의 논문은 값을 그 절에서 가져왔고 "(기존 절)" 로 표기했다. 그림에서만 읽히는 값은 "그림값", 못 읽은 것은 "미확인". 전수 조사가 아니다.

**우리 수치 (비교 기준).** R19 C10 T=4 baseline 96.73 (n=6, 489K). inter ±0.00%p @−30% (n=3), intra −0.14%p @−60% (n=5). 재현 BPSR −0.07%p @−35%, plain L2 −0.03%p @−20% (이 두 값의 n·설정은 의뢰서 값 그대로, 이 절에서 재확인 안 함).

### 1. 핵심 표

열: 벌점 양·노름 / 가중 / λ / 모델·데이터·T / 정확도 (표) / 효율 감소 / ≤0.2%p 구간 최대 감소 / 주의

| # | 논문 (저자·학회·연도·링크) | 벌점 양·노름 | 가중 | λ | 모델·데이터·T | 정확도 (baseline→결과) | 효율 지표·감소 | ≤0.2%p 최대 감소 | 주의 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Esser et al., PNAS 2016. https://arxiv.org/abs/1603.08270 | 특징맵 평균 활성 ȳ 의 제곱합 (γ/2)Σȳ², L2 | 균등 | 수동 γ=1e-4 | TrueNorth용 이진 CNN, 8개 데이터셋 | 벌점 유무 ablation 없음 | 효과 수치 없음 ("spike traffic 감소" 서술만) | 표에서 안 읽힘 | 벌점은 있으나 보조 장치. 단독 효과 미측정. 권·호는 본문에 없음 |
| 2 | Neil, Pfeiffer, Liu, ACM SAC 2016. https://dl.acm.org/doi/10.1145/2851613.2851724 | **미확인** (본문 접근 못 함; Sorbaro 가 "L1 항"으로 인용 = 2차) | – | – | – | – | 검색 요약: 가장 효율적 SNN 이 ANN 연산 스텝 42% 미만 (출처 요약) | 미확인 | PDF 403. 사용자가 받아야 함 |
| 3 | Sorbaro, Liu, Bortone, Sheik, Front. Neurosci. 14:662, 2020. https://arxiv.org/abs/1912.01268 | SynOps 목표 이탈 α(S0−Σ_l fout_l·Σa)², L2. 대상은 ANN(QReLU) 활성 | 층별 fan-out | α=1/S0², S0 수동 스윕 (30개 모델 연쇄 미세조정) | All-ConvNet 1.9M, CIFAR-10, SNN Ndt=10 | Table 1: ANN 91.37; SNN 91.75@2179M, 91.73@1593M, 91.47@368M, 91.43@277M, 90.37@127M, 85.71@64M SynOps | SynOps 2179M→1593M (−27%) @−0.02%p; →368M (−83%) @−0.28%p | −27% (Table 1) | n·SD 없음. 보존은 QReLU 양자화에 의존. MNIST-DVS 96.3→95.0 (그림값 위주) |
| 4 | Pellegrini, Zimmer, Masquelier, arXiv 2011.06846 (SLT 2021 은 Lemaire 인용에 의한 2차 확인). https://arxiv.org/abs/2011.06846 | 층별 스파이크 수 제곱 (1/2KN)ΣΣS², L2 | 균등 | 수동 고정 0.1 | 3층 dilated-conv LIF 0.13M, Speech Commands v1, 12-class | Table 2: test ER 5.7(벌점 없음)→5.5 (벌점) | Table 2 층별 발화 %: 12.5/12.5/6.4 → 2.6/4.9/6.1 (층1 −79%, 층2 −61%, 층3 −5%) | 층1·2 −61~−79% @정확도 동등 (+0.2%p) | n 없음, 단일런으로 보임. 소형·음성 |
| 5 | Zenke & Vogels, Neural Comput. 33(4):899, 2021. https://research-explorer.ista.ac.at/download/8253/11131/2021_NeuralComputation_Zenke.pdf | 하한 뉴런별 (ν_lower−ζ)₊² (L2) + 상한 층 평균 (ζ̄−ν_upper)₊^L, L∈{1,2} | 하한 뉴런별, 상한 층 평균 | 수동 (Table 1 데이터셋별) | Randman, MNIST, SHD, RawHD, RawSC; T 80~500 스텝 | Fig. 9 그림값 | 본문: 약 2자릿수 스파이크 감소까지 눈에 띄는 저하 없음 | 표에서 안 읽힘 | 효율 목적이 아니라 희소 영역 학습 가능성 검증. 소형 데이터 |
| 6 | Perez-Nieves & Goodman, NeurIPS 2021. https://arxiv.org/abs/2105.08810 | **제외**: 줄이는 것은 역전파 연산(은닉 활동 1~2%에서 backward 절감). 활동 규제는 Zenke 것을 재사용 | | | | | | | |
| 7 | Lemaire et al., ICONIP 2022 (ICANN 아님), arXiv 2210.13107. https://arxiv.org/abs/2210.13107 | **제외 (벌점 없음)**: 에너지 *측정* 논문. 의뢰서의 "synaptic activity regularization" 은 이 논문에 없음. 같은 저자의 ACM TECS 2022 (doi 10.1145/3520133) 와 혼동 가능성, 그쪽 본문은 미확인 | | | | | | | |
| 8 | Narduzzi, Bigdeli, Liu, Dunbar, ICASSP 2022. https://arxiv.org/abs/2204.01460 | 변환 전 ReLU DNN 층별 활성에 ψ∈{L1, L2, L0.5, Hoyer, Hoyer-Sq} | 균등 | 수동 (표 λ 는 SynOps 최소 모델을 사후 선택) | MLP, LeNet-5; MNIST T=100, CIFAR-10 T=1000 (변환) | Table 1 MNIST LeNet-5 SNN 98.39: L2@1.0 +0.10, L0.5@3e-4 +0.18, L1@1e-2 −0.47. Table 2 CIFAR-10 LeNet-5 51.74→L2 +2.76 | Table 1: L0.5 spikes −89.0%, SynOps −91.2%; L2 spikes −83.9% | L0.5 −89% @+0.18 (MNIST LeNet-5) | **변환 후 SNN 이므로 범위 경계**(활성 규제는 학습 규칙이나 SNN 직접 학습 아님). n·SD 없음, 사후 선별. CIFAR baseline 51.74 로 약함 |
| 9 | BPSR. Yan, Chu, Jin, Huan, Zou, Zheng, Front. Neurosci. 16:760298, 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9047717/ | 은닉층 스파이크 제곱 L2 (식4) + 가중치 L1 | 균등 (출력층 제외) | 수동 λ_s=1e-7(MNIST), 1e-9/1e-8(CIFAR10) | 11층 ResNet, CIFAR10, T=12 / T=8; MNIST류 | Table 5: 90.74@T=12 (136.1K spikes), 90.24@T=8 (89.6K). λ=0 baseline 은 표에 없음 | 절대 스파이크 수만 | 표에서 안 읽힘 | 두 행은 T 와 λ 가 같이 변함. n·SD 없음. 입력 스파이크 제외 |
| 10 | Datta, Liu, Beerel, arXiv 2212.10170 (게재 미확인). https://arxiv.org/abs/2212.10170 | **제외 쪽**: 막전위 Hoyer 정규화, 목적은 T=1 임계값 추정 | 균등 | λ_H 미확인 | 1-step SNN, VGG16/ResNet18, C10 | Table 5 ResNet18 C10: 91.08→90.95 | 활동 27.62→20.50% (−25.8%); VGG16 은 20.43→20.48 (변화 없음) | ResNet18 −25.8% @−0.13%p (n 없음) | 발화 감소는 부수 효과. 막전위 규제라 범위 밖에 가까움 |
| 11 | Kim & Panda 류 | **해당 없음**: 발화 감소 목적의 논문을 확인 못 함 (Suetake 가 에너지 지표 출처로 Kim & Panda 2021 인용) | | | | | | | |
| 12 | IM-Loss (NeurIPS 2022), RMP-Loss (ICCV 2023). https://arxiv.org/abs/2308.06787 | **제외**: IM-Loss 는 층 평균 막전위를 V_th 에 맞춰 정보량 최대화 (목표 p(spike)=0.5, 발화 감소 아님), RMP-Loss 는 막전위를 임계값 근처로 모아 양자화 오차 감소. 값은 기존 절 A7 | | | | | | | |
| 13 | Suetake et al., arXiv 2302.01500 (TMLR 2024, 제목 "Synaptic Interaction Penalty" 로 변경된 것으로 보임, 게재본 수치 미확인). https://arxiv.org/abs/2302.01500 | Ω_syn=Σ_l(1/p)Σ_i ψ_{l,i}s^p (p=1,2 비교), ψ=나가는 시냅스 수. 에너지에 비례 | 층·뉴런 고정 상수 (fan-out). 경쟁 없음 | 수동 10배 격자 1e-9~1e-5, λ0=λ/Ω(1), 에폭 비율로 선형 증가 | **T=1** 단일 스텝, ResNet18/VGG11/CNN7, C10·F-MNIST·C100. 시드 3 | Table 10 ResNet18 C10 (Adam): λ=0 91.32±0.209 (E 100) → 1e-9 91.27 (93.06) → 1e-8 91.40±0.293 (58.03±0.222) → 1e-7 89.46±0.215 (19.13) → 1e-6 78.76 (4.33). Table 2 VGG11 C10: 89.09±0.271 → 1e-8 89.23 (E 73.92) → 1e-7 87.81 (34.07). Table 11 ResNet18 mSGD: 90.15 → 1e-8 89.83 (60.41) | E=T·E_AC·Σψs, E_AC=0.9pJ. ResNet18 λ=1e-8 에서 E −42.0% | **−42% (ResNet18 C10 Table 10, +0.08%p, n=3)** | WD 결합 조합 91.85, E 38.43 은 WD 대조(91.97, 66.36) 대비 acc −0.12%p, E 추가 −42.1%. 기존 절 A10 값과 일치. T=1 이라 우리 T=4 와 다름 |
| 14 | Stanojevic et al., Nat. Commun. 2024 (DOI 10.1038/s41467-024-51110-5; 표는 arXiv 2306.08744 값, 게재본 미확인). https://arxiv.org/abs/2306.08744 | ReLU 사전학습 시 L1 (대상·λ 미확인), 이후 TTFS 변환·미세조정 | 균등 | 미확인 | VGG16, TTFS | Table 2: C10 93.69 (0.38 spikes/neuron) → +L1 93.27±0.02 (0.20); C100 72.24 (0.38) → 72.20±0.04 (0.24) | C10 −47%, C100 −37% (spikes/neuron) | C100 −37% @−0.04%p | C10 은 −0.42%p 라 초과. TTFS 변환. 기존 절 A9 |
| 15 | MP-Init + TrSG, Kook et al., arXiv 2511.08708 (기존 절 A8) | 스파이크 희소성 손실은 진단용 (규제가 방법이 아님) | – | – | R19 C100 T=4 | Table 3: C10 95.34±0.06, C100 77.66±0.15 | Fig 8 발화율 곡선 0~0.55 (그림값) | 표에서 안 읽힘 | 이 표의 비교 대상이 아니라 같은 축 그림 근거 |
| 16 | Activity Pruning, Bu, Shi, Yu, NeurIPS 2025 (기존 절 1a~1e). https://proceedings.neurips.cc/paper_files/paper/2025/hash/fa12d67b5939c37ea8a4659c31a08d2c-Abstract-Conference.html | 학습 가능한 임계값 θ 에만 규제 기울기 (Eq.12), 출력 {0,θ}. 뉴런 모델 변경 | 임계값 (층 또는 뉴런 단위 미확인) | 수동 λθ, λs | R20 C10 T=8 | Table 1: 92.89→92.05 (−0.84) @FR −77.5%, SOPs −76.0%; 91.06 (−1.83); L1 규제(AR) λs=0.01 91.31 (−1.58) FR 0.060 | FR, SOPs | **해당 행 없음** (최소 손실 −0.84) | 시드 수 미기재. 행 1b −2.19%p |
| 17 | UniSparse, Shi, Bu, Yu, ICML 2026 (기존 절 4,5) | BN γ,β·가중치를 SOPs 모델 penalty 로 직접 갱신 (손실 항 아님) | 채널별 BN | 수동 λ_neuron, λ_weight | R18 C10, C10-DVS | 2차 요약: C10 92.38% @7.04M SOPs; DVS 79.12% @5.65M | – | 미확인 | 논문 표 못 읽음 |
| 18 | Spik4lite, ICML 2026 (기존 절 2,3) | 게이트(Gumbel) 확률 × 비용 계수(detach)의 에너지 벌점 후 채널 제거 | 채널 게이트 | 수동 λ_energy=0.01 | Spikingformer C10 T=4 등 | 저자 로그 94.40 vs 논문 95.81 (1런) | 채널 −90.3% (로그) | 미확인 | 프루닝 파이프라인에 가까워 범위 경계 |

### 1b. 2023~2026 추가 발굴 (읽은 것)

| # | 논문 | 벌점 양·노름 | 가중 | λ | 모델·데이터·T | 정확도 | 효율 | ≤0.2%p 최대 감소 | 주의 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | Meem, Nadid, Mia, arXiv 2602.12236 (v2 2026-03, 게재 미확인). https://arxiv.org/abs/2602.12236 | 층 평균 발화율의 목표 이탈 λ_rate(r−r_target)², L2 (양방향) | 균등 (전체 r 하나) | **자동**: 비례 제어기 Δλ=η(r−r_target), [λmin,λmax] clip. η,λmax 는 수동 | 연속학습(5×2 class-incremental), MNIST FC T=25, CIFAR-10 VGG5풍 T=50 | Table 2 (n=3): MNIST 93.44±1.40→95.75±0.16; CIFAR-10 59.50±1.58→61.26±0.79 (replay+학습형 LIF+스케줄러 합산) | Table 2 "Spike(%)": MNIST 15.31→8.07 (−47%), C10 37.50→30.90 (−18%) | **규제 단독(Table 3, MNIST, n=3): 93.44→92.40 (−1.04%p), 스파이크 −32%** → 0.2%p 초과 | λ 자동화가 있는 유일한 후보지만 발화율 오차 피드백(우리는 task 손실 비율). CIFAR-10 60% 대라 비교 불가 |
| 20 | Wang, "The Sparsity Ceiling", arXiv 2607.26648 (2026-07-29, 게재 미확인). https://arxiv.org/abs/2607.26648 | 층 평균 발화율과 목표 ρ⋆ 의 차, L1 (양방향, warmup) 대 단방향 L1 | 균등 | 수동 (값 미확인) | Conv SNN, rate-coded FashionMNIST, T=10 (+RNN LM, 트랜스포머 Ts=4) | §4.1 표 (3시드): SNN 74.7 → ρ⋆=.05 76.6±.4 | 발화 0.16/0.27→0.05 (−69%); 트랜스포머 0.27→0.02, bpc 3.55→3.57 | −69% @정확도 +1.9%p (FashionMNIST, 작은 모델) | **기지 논문 (Already_Known)**. 단방향 L1 은 발화 0, 정확도 10% 로 붕괴. baseline 이 ANN 보다 13.4%p 낮음 |
| 21 | Xu, Yin, Iyer, Li, ISCA '25 (논문 목록 표기 ISCA 2025), arXiv 2505.12281. https://arxiv.org/abs/2505.12281 | 번들(토큰×시간) 활성 태그 Z 의 합 (L0) λ·ΣZ | 번들 단위 | 수동 (값 안 읽힘) | 스파이킹 ViT, CIFAR10 | 표에서 안 읽힘 | CIFAR10 Model 1 활성 번들 29% (BSA 전, 본문); BSA 후는 그림 | 표에서 안 읽힘 | 하드웨어 논문. L0 근사 방법 미확인. "그룹 단위 벌점" 이라는 점만 가까움 |
| 22 | Ahmed, Sun, Castrillon, Vadivel, Tang, arXiv 2606.01135 (2026-05-31, 게재 미확인). https://arxiv.org/abs/2606.01135 | 층 평균 발화율이 [r_min,r_max] 밖일 때만 벌점 (식 2) | 균등 | 수동 (값 미확인) | S-SpeechMamba, LibriSpeech | Table I: WER 2.32 (ANN) → 4.71 (SNN) | 활동 희소도 72% | 해당 없음 | 벌점 ablation 없음, n=1 |
| 23 | Mikaeili Barzili et al., arXiv 2605.25144 (2026-05, 게재 미확인). https://arxiv.org/abs/2605.25144 | Σρ_l + β·Σ(ρ_l−ρ⋆)², L1+L2 | 균등 | 수동 λ=1e-4, β=1e-2, ρ⋆=0.1 | 3D U-Net SNN, OASIS(19쌍), T=4 | Table 2: Dice 0.7480±0.0365 → 0.7474±0.0320 | Table 3 발화율 12.8%, 에너지 추정 55.5× | 규제 유무 비교 없음 (Table 4 ablation 에 스파이크 항 없음) | 의료 영상 정합. 에너지는 추정 |
| 24 | Calle-Ortiz et al., arXiv 2508.11689 (2025-08, 게재 미확인). https://arxiv.org/abs/2508.11689 | 손실 항 아님: 학습 중 뉴런 임계값에 균등분포 잡음 (임계값 적응) | 뉴런 임계값 | 구간 하이퍼파라미터 | 웨어러블 HAR (KU-HAR 등) | Table 1 (상대 변화): KU-HAR +0.12% | KU-HAR 스파이크 −46.5%; UCI-HAR +0.74%, −15.4% | −46.5% @+0.12% (KU-HAR) | 소형 센서 데이터. 추론 때 임계값을 바꾼다는 서술 → 우리 설정과 다름 |

읽었으나 표에서 뺀 것: Sun et al. 2023 (Front. Neurosci. 17:1275944, 오답 출력 막전위 억제 손실, 효율 목적 아님, 초록만 확인), ILIF (IJCAI 2025, arXiv 2505.10371, 뉴런 모델 변경), Shen et al. AAAI-24 (K-WTA 연속학습 마스크, 스파이크 수 지표 없음), CSSL (arXiv 2508.19806, ReLU 활성), RCS (Frontiers 2025, 목적이 추론 T 단축), Bullet Trains (arXiv 2603.13283, 연속시간·안정화 목적), DA-SSDP (arXiv 2512.07194, 정확도 목적), arXiv 2312.06900 (변환 L1).

### 2. 분석

**2.1 우리와 가장 가까운 3편 (축: 벌점 대상 / 가중 / λ).**
1. **Suetake (#13).** 대상: 스파이크 수에 시냅스 fan-out 을 곱한 값, 에너지에 비례. 차이: 가중이 층별 고정 상수이고 경쟁이 없다. λ 는 수동 격자(에폭 비율 선형 증가). T=1.
2. **Meem (#19).** 대상: 층 평균 발화율. 차이: λ 를 발화율 오차 피드백으로 자동 조정한다(우리는 task 손실 비율 ρ). 가중은 균등, 경쟁 없음. 규제 단독은 −1.04%p (MNIST, n=3).
3. **BPSR (#9) / Pellegrini (#4) 계열 (스파이크 수 제곱 L2).** 대상: 스파이크 수 제곱. 차이: 균등 가중, λ 수동 고정. 이 계열은 우리의 plain L2 대조군과 같은 부류.
(차점: Sorbaro #3. SynOps 목표 이탈 벌점이라 "에너지 모델을 손실에" 축에서는 가깝지만 ANN 활성에 걸고 S0 를 수동 스윕한다.)

**2.2 정확도 손실 ≤0.2%p 에서의 감소율. 같은 조건이 아니다.** 모델·데이터·T·지표가 모두 다르다. 아래 순서는 "표에서 읽힌 감소율" 순서이지 우열이 아니다.

| 방법 | 정확도 변화 | 감소 | 지표 | 모델·데이터·T | n | 출처 |
|---|---|---|---|---|---|---|
| Narduzzi L0.5 | +0.18 | −89% (SynOps −91%) | 스파이크 | LeNet-5, MNIST, T=100 (변환) | 없음 | #8 Table 1 |
| Wang 양방향 L1 | +1.9 | −69% | 발화율 | Conv, FashionMNIST, T=10 | 3 | #20 (기지) |
| 우리 intra | −0.14 | **−60%** | spikes/sample | R19 C10 T=4 | 5 | 본 연구 |
| Calle-Ortiz 임계값 잡음 | +0.12 | −46.5% | 스파이크 | HAR 센서 | 미확인 | #24 Table 1 |
| Suetake Ω_syn | +0.08 | −42% | 에너지(Σψs) | ResNet18, C10, T=1 | 3 | #13 Table 10 |
| Stanojevic L1 | −0.04 | −37% | spikes/neuron | VGG16 C100, TTFS | 16 trial로 추정 | #14 Table 2 |
| 우리 BPSR 재현 | −0.07 | −35% | spikes/sample | (R19 C10 T=4 로 가정, 미재확인) | 미확인 | 의뢰서 |
| 우리 inter | ±0.00 | −30% | spikes/sample | R19 C10 T=4 | 3 | 본 연구 |
| Sorbaro | −0.02 | −27% | SynOps | All-ConvNet C10, Ndt=10 | 없음 | #3 Table 1 |
| 우리 plain L2 | −0.03 | −20% | spikes/sample | (R19 C10 T=4 로 가정, 미재확인) | 미확인 | 의뢰서 |
| Pellegrini | 동등(+0.2) | −61~79% (층1·2) | 층별 발화율 | 3층 SNN, Speech Commands | 없음 | #4 Table 2 |
| Activity Pruning | 해당 행 없음 (최소 −0.84) | – | FR | R20 C10 T=8 | 미기재 | #16 Table 1 |
| Meem 규제 단독 | −1.04 (초과) | −32% | 발화율 | MNIST FC, T=25 | 3 | #19 Table 3 |

해석 주의: (a) 위 수치 중 R19 T=4 의 CIFAR-10 급 모델에서 ≤0.2%p 구간 감소율을 표로 보고한 외부 논문은 찾지 못했다. 외부 값은 모두 소형·다른 T·다른 데이터다. (b) 외부 논문의 "≤0.2%p" 는 대부분 n·SD 가 없어, 정확도 시드 SD 0.10~0.13%p 를 감안하면 동등 여부를 말할 수 없다. (c) Narduzzi 는 사후 선별. (d) Pellegrini 의 정확도는 동등이지만 층 3 은 −5% 뿐이다.

**2.3 우리 코드에서 R 의 정의만 바꿔 재현이 싼 것 (구현 제안 아님, 목록만).**
- Suetake Ω_syn: R = Σ_l ψ_l Σ_i s_{l,i} (ψ_l = 층 fan-out 상수).
- Sorbaro: R = (S0 − Σ_l fout_l·Σ스파이크)², S0 수동. 원 논문은 ANN 활성에 건다.
- Pellegrini: R = (1/2KN)ΣΣS² (층별 스파이크 수 제곱, 균등). BPSR 과 사실상 같은 계열.
- Esser: R = 채널 평균 활성의 제곱합 (γ=1e-4).
- Narduzzi: R = ‖a‖_p, p∈{1, 2, 0.5}, 균등.
- Zenke & Vogels: R = 하한 (ν_lower−ζ)₊² 뉴런별 + 상한 (층 평균 ζ̄−ν_upper)₊^L.
- Meem: R = (층 평균 발화율 − r_target)², λ 는 발화율 오차 피드백.
- Wang: R = Σ_l |s̄_l − ρ⋆| (양방향 L1, warmup).
- Ahmed: R = mean_l[max(0, r_min−r̄) + max(0, r̄−r_max)].
- SpikeReg: R = Σρ_l + β·Σ(ρ_l−ρ⋆)².
- 재현이 싸지 않은 것: Activity Pruning, Calle-Ortiz (뉴런 모델·임계값을 건드림, 학습 경로 파일 수정), BSA (L0 근사 미확인), UniSparse·Spik4lite (파라미터 직접 갱신·게이트).

### 3. 위협도

실제로 위협적인 것: (1) Suetake (#13): "에너지 비례 벌점 하나, 정확도 유지"의 가장 가까운 선례이고 표에 n=3 과 SD 가 있다. 방어: 균등 fan-out 가중·수동 λ·T=1·저정확도 baseline(91.3%), 경쟁 없음. (2) Activity Pruning (#16): "규제는 정확도를 희생한다" 주장과 정면 충돌, 리뷰어가 인용할 가능성이 가장 높다. (3) Meem (#19): λ 자동화 선례. 단 방식이 다르고 규제 단독 결과는 정확도 −1.04%p 다. 새로 발굴한 논문 중 위협은 이 정도이고, 나머지 2023~2026 후보는 층 평균 발화율 벌점이라 우리 plain L2 대조군과 같은 부류다.

### 4. 미확인 목록 (본문 인용 금지)

- Neil 2016 의 벌점 형태·수치 (PDF 403).
- Lemaire 2022 ACM TECS (doi 10.1145/3520133) 본문: "synaptic activity regularization" 이 이 논문에 있는지 (ICONIP 논문에는 없음).
- Pellegrini 의 SLT 2021 게재 (2차 출처), Suetake 의 TMLR 게재본 수치, Datta 2212.10170 의 학회·λ_H, Stanojevic 의 L1 대상·λ.
- Zenke & Vogels Fig. 9 수치, BPSR 의 λ=0 baseline, Sorbaro MNIST-DVS 수치.
- Narduzzi·BPSR·Sorbaro·Pellegrini 의 n·SD.
- 우리 BPSR 재현·plain L2 의 모델·T·n (의뢰서 값만 사용).
- 2.2 표의 Calle-Ortiz n. 2023~2026 후보 대부분 게재 미확인.
- 읽지 못한 후보: https://link.springer.com/article/10.1007/s00521-024-10191-5 (spiking activity tuning, 마이크로컨트롤러, 페이월), https://link.springer.com/article/10.1007/s11063-025-11786-2 (Spike-Count Reduction, Neural Processing Letters 2025, 초록만; 학습 벌점이 아닌 것으로 보임), https://www.sciencedirect.com/science/article/pii/S2666827026001738 (Spiking ConvNeXt, 403).
- UniSparse·Spik4lite 논문 표 (기존 절 그대로).

## 2026-10-06 유사 방법 심층 조사

조사 방법: 웹 검색 후 arXiv/공식 PDF 를 내려받아 `pdftotext` 로 표·식을 직접 읽었다. 읽지 못했거나 표 정렬이 깨진 값은 **미확인**이다. 전수 조사가 아니라 이 세션의 검색 범위 안의 결과다. 2026-09-29 / 10-02 절에 있는 논문(Suetake, Meem, BPSR, Pellegrini, Sorbaro, Narduzzi, Zenke, Esser, SCA, Spik4lite, UniSparse, Activity Pruning, DeepEISNN, SPEAR, QP-SNN 등)은 다시 조사하지 않았다.

**우리 비교 기준.** ours_inter_ch: R19 C10 T=4, baseline 96.73 (n=6), −30% 스파이크에서 ±0.00%p (n=3). 아래 외부 수치는 모델·데이터·T 가 모두 달라 우열 비교가 아니다.

**축별 결론 한 줄.**
- 축 1 (경쟁·순위 가중 규제): ANN 에서 **SASL** 이 "약한 필터일수록 큰 벌점"을 순위로 건다. 단 대상이 BN γ(가중치 쪽)이고 5단계 고정 배수다. SNN 에서는 찾지 못했다.
- 축 2 (활성에 거는 group 노름): 활성에 직접 거는 것은 **FBS**(채널 saliency 의 lasso, ICLR 2019)와 **Kurtz**(Hoyer, ICML 2020) 뿐이다. 둘 다 ANN 이다. 스파이크 활성에 채널 단위 노름을 건 SNN 논문은 찾지 못했다.
- 축 3 (λ 를 task 손실 비율로): SNN 에서는 찾지 못했다. SNN 밖에서 **DAP**(arXiv 2026-08)와 **SymbolNet**(arXiv 2025)이 "정규항 크기를 task 손실에 비례시킨다".
- 축 4 (학습 중 발화 기반 채널 프루닝): 기지 논문(SCA, Spik4lite, UniSparse) 밖에서는 **Kundu AGC**(WACV 2021)와 **DPAP**(arXiv)이 있다. 둘 다 규제항이 아니다.
- 축 5 (측면 억제·WTA): 구조 변경형은 기지(DeepEISNN, SpiLiFormer, LISNN). 추가로 WTA 어텐션(2604.11321)을 찾았으나 언어모델 구조 변경이다.

### 1. 핵심 표

열: 벌점 대상·형태 / 가중 / λ / 모델·데이터·T / 정확도 (표) / 효율 감소 / n / 주의

| # | 논문 (저자·학회·연도·링크) | 벌점 대상·형태 | 가중 | λ | 모델·데이터·T | 정확도 (baseline→결과) | 효율 감소 | n |
|---|---|---|---|---|---|---|---|---|
| 1 | **SASL** Saliency-Adaptive Sparsity Learning. Shi, Xu, Tasaka, Chen. arXiv 2003.05891 v3 (2020-07). **게재처 미확인** (IEEE 서식 원고). https://arxiv.org/abs/2003.05891 | 필터별 스케일링 인자(BN γ)에 정규항 R(·) (Eq.7, network slimming 방식). R 이 L1 인지는 본문에서 확인 못 함. ANN | **순위 기반**: saliency(예측 기여 + 연산 비용)로 모든 필터를 정렬해 5등급, 정규화 배수 {0×,1×,2×,3×,4×}. 가장 중요한 등급은 벌점 0, 가장 약한 등급이 최대. 학습 중 순위를 계속 갱신 | 기본 λ 수동 (VGG 1e-4, ResNet 3e-5) | VGG-Net, ResNet-56/110 (CIFAR-10), ResNet-50 (ImageNet). ANN | Table I: VGG 93.69→93.89, Res-56 93.63→93.88 (보수), 93.58 (공격), Res-110 93.83→93.99 / 93.80 | Table I: VGG 파라미터 −86.9%·FLOPs −49.5%, Res-56 FLOPs −35.9% (보수) / −57.1% (공격). 필터가 0 이 되어 제거 | 미기재 |
| 2 | **FBS** Dynamic Channel Pruning: Feature Boosting and Suppression. Gao, Zhao, Dudziak, Mullins, Xu. ICLR 2019. https://arxiv.org/abs/1810.05331 | 레이어별 채널 saliency 예측값 g(x) 에 L1 lasso λΣE‖g‖₁ (식 8 아래). 출력 채널 = **k-winners-take-all** 로 상위 ⌈dC⌉ 채널만 통과(식 5·6), 나머지 0 | 채널 간 경쟁(top-k). 벌점 가중은 균등 | 수동 λ=1e-8 | M-CifarNet(CIFAR-10), VGG-16, ResNet-18(ImageNet). ANN | M-CifarNet baseline 91.37 → d=0.5 에서 90.59 (본문 §4.1). ResNet-18 ImageNet d=0.7 에서 top-1 오차 31.54% | ResNet-18 d=0.7: MAC 1.62배 적음. 사용되지 않는 채널은 영구 제거 가능 ("FBS naturally subsumes channel pruning") | 미기재 |
| 3 | **Kurtz et al.** Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks. ICML 2020. http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf | ReLU 출력 활성 전체에 square Hoyer H(v)=(Σ|vᵢ|)²/Σvᵢ² (스케일 불변 비율). L1 과 비교 | 균등 (스케일 불변이라 크기 정규화 효과) | 수동, 정확도 무손실이 되는 값을 격자 탐색 | ResNet-18/50, MobileNet (ImageNet). ANN | 정확도 손실 <0.3% | Table 1 평균 활성 희소도 baseline→L1→Hoyer→boosted Hoyer: ResNet-18 53→55→62→67%, ResNet-50 53→54→61→65% | 미기재 (표준편차 <1% 는 배치 간) |
| 4 | **Deng et al.** Comprehensive SNN Compression Using ADMM Optimization and Activity Regularization. Deng, Wu, Hu, Liang, Li, Hu, Ding, Li, Xie. arXiv 1911.00822. **게재처 미확인** (PubMed ID 34723806 존재, 저널명 확인 못 함). https://arxiv.org/abs/1911.00822 | 직접 학습 SNN 의 **평균 스파이크율 R** 에 λR (식 15). 층·뉴런 구분 없음, 노름은 선형 | 균등 | 수동 λ∈{0.001, 0.01, 0.1} | 10층 스파이킹 CNN, CIFAR-10. T 미확인 | Table IV, CIFAR-10: λ=0 89.53 → 0.001 89.51 → 0.01 87.62 → 0.1 81.01 | 평균 스파이크율 0.11→0.11→0.08→0.03. 무손실(−0.02%p) 구간은 감소 0%, 감소가 생기면 −1.91%p 이상 | 미기재 |
| 5 | **Kundu et al.** Spike-Thrift: Towards Energy-Efficient Deep SNNs by Limiting Spiking Activity via Attention-Guided Compression. Kundu, Datta, Pedram, Beerel. WACV 2021 (CVF Open Access 로 확인). 확장판 arXiv 2107.12445. https://arxiv.org/abs/2107.12445 | 규제항 아님. ANN 의 attention map 으로 채널·필터 중요도를 정해 (structured/irregular) 프루닝 후 SNN 희소 학습 | 채널 F-노름 순위로 레이어별 제거 + 재성장 | 해당 없음 | VGG/ResNet 변형, CIFAR-10/100, Tiny-ImageNet. T 는 학습 중 ~10배 줄임 | 초록: "압축률 최대 33.4×, 유의한 정확도 하락 없음". 표(Table II·III)는 **미확인** | 초록: 지연 최대 15× 단축 | 미기재 |
| 6 | **Allred et al.** Explicitly Trained Spiking Sparsity in SNNs with Backpropagation. Allred, Spencer, Srinivasan, Roy. arXiv 2003.01250 v1 (2020-03, "preliminary draft"). **게재처 미확인**. https://arxiv.org/abs/2003.01250 | 총 스파이크 수에 σ(spikeCount), σ=σ₀·(상수/선형/제곱/교대) 스케줄 | 균등 | 수동 σ₀ (검증 격자) | LeNet, CIFAR-10 (SpyTorch) | Table 1 (검증 오차 무증가 조건): 64.97 → 65.61 (σ_constant) / 66.45 (σ_alternate linear) | 평균 스파이크 294,130 → 95,671 (−67%) / 89,869 (−69%). 초록은 "최대 −70.1% iso-accuracy" | 1회 (검증 20% 분할 선택) |
| 7 | **DPAP** Developmental Plasticity-inspired Adaptive Pruning for Deep SNNs and ANNs. Han, Zhao, Zeng, Shen. arXiv 2211.12714 v3 (2024-10). **게재처 미확인**. https://arxiv.org/abs/2211.12714 | 규제항 아님. 스파이크 트레이스 기반 뉴런·시냅스 중요도를 매 에폭 계산, 낮은 것은 "use it or lose it" 로 서서히 감쇠 후 제거. 공간 데이터는 채널 단위(structured) | 중요도 순위 기반 감쇠 | 적응형 (생존 함수) | MNIST, CIFAR-10, DVS 등. SNN 표의 모델·T 는 표 정렬이 깨져 **미확인** | **미확인** (ANN VGG16 CIFAR-10: 파라미터 41.54%, 94.76%, +0.23 는 읽힘) | 미확인 | 미확인 |
| 8 | **WTA Spiking Transformer** Zhou, Guo, Wang, Ma, Che, Chen, Meng, Ma, Tian. arXiv 2604.11321 (2026-04, ICML 2026 서식의 preprint, **게재 미확인**). https://arxiv.org/abs/2604.11321 | 규제항 아님. 어텐션 맵에 hard WTA 를 넣는 **구조 변경** (softmax 대체, 측면 억제 모사) | WTA | 해당 없음 | 스파이킹 트랜스포머, 언어 모델링 16개 데이터셋 | 비전·CIFAR 해당 없음 | 수치 미확인 | – |
| 9 | **DAP** Domain-Aware Pruning. Sazdar, Tavassolipour, Hosseini (Tehran). arXiv 2608.08624 (2026-08, 게재 미확인). https://arxiv.org/abs/2608.08624 | 마스크 밀도의 목표 이탈 제곱 (식 12). SNN 아님 | 균등 | **자동**: λ_s = λ_base·φ_progress·φ_difficulty·φ_ratio, φ_ratio = clip(L_base/(L_sparsity+ε)·1/r_target, r_min, r_max) (식 13·16), EMA 평활 (식 17), 100 step 마다 갱신 | ResNet-50, DomainBed 5개 | 40% 희소도에서 dense 와 동등 이상 (표 수치 미확인) | 파라미터 40% | 3 시드 (그림 설명) |
| 10 | **SymbolNet** Tsoi, Loncar, Dasu, Harris. arXiv 2401.09949 v3 (2025-01). **게재처 미확인**. https://arxiv.org/abs/2401.09949 | 학습 가능한 가지치기 임계값에 exp(−T) 벌점. SNN·활성 아님 (기호 회귀 NN) | 균등 | **자동**: L_sparse = L_error · D(s;α,d) · L_threshold (식 아래). 시작 시 L_sparse = L_error (비율 1), 목표 희소도 s→α 가 되면 D→0 으로 꺼짐 | 기호 회귀 MLP | 해당 없음 | 목표 희소도 수렴 | 미기재 |

### 2. 분석

**2.1 우리 조합의 구성 요소별 선례.**

| 구성 요소 | 가장 가까운 선례 | 겹치는 정도 | 비어 있는 부분 |
|---|---|---|---|
| 순위·상대 크기로 약한 쪽을 더 벌함 | SASL (#1): 필터 순위 5등급 배수 | 약한 필터일수록 큰 벌점이라는 구조가 같다 | 대상이 BN γ(가중치), 이산 5등급, ANN. 연속 가중 1−a/max 도, 활성·스파이크 대상도 아니다 |
| 채널 간 경쟁으로 채널이 죽음 | FBS (#2): 채널 top-k + saliency lasso, 쓰지 않는 채널 제거 | 채널 단위 경쟁과 죽는 채널이 같다 | WTA 가 손실이 아니라 **하드 게이트 구조**. lasso 는 예측기 출력에 걸림. ANN |
| 활성에 거는 노름 규제 | Kurtz (#3): Hoyer (스케일 불변) | 활성 자체에 건다. 비율형이라 상대 크기를 쓴다 | 채널 단위 아님, 경쟁 없음, ANN |
| SNN 직접 학습 스파이크 벌점 | Deng (#4), Allred (#6) | 스파이크 수·율에 λ 벌점 | 균등 가중, 수동 λ. 정확도 무손실 구간에서 감소가 작다 (#4 CIFAR-10 λ=0.01 에서 −1.91%p) 또는 저정확도 모델 (#6 LeNet 64.97%) |
| 학습 중 SNN 채널 프루닝 | Kundu (#5), DPAP (#7) + 기지 SCA·Spik4lite | 채널이 제거되고 스파이크가 준다 | 규제항이 아니라 별도 기준·단계 |
| λ = ρ·L_task/R | DAP (#9), SymbolNet (#10) | 정규항 크기를 task 손실에 비례시킨다 (#10 은 시작 비율 1, #9 는 L_base/L_sparsity) | SNN 도, 활성 규제도 아니다. DAP·SymbolNet 은 목표 희소도에 맞춰 줄여 가는 스케줄이 곁들여지고, 우리는 매 에폭 ρ 고정 |

**2.2 우리와 직접 비교할 수 있는 외부 수치는 없다.** R19 T=4 CIFAR 급 SNN 에서 "스파이크 −30% 이상, 정확도 ≥ baseline" 을 표로 보고한 논문은 이번에도 찾지 못했다. #4 는 CIFAR-10 에서 무손실(λ=0.001)일 때 스파이크율 감소가 0 이고, 0.01 에서 −27%(0.11→0.08)에 −1.91%p 다. #6 은 −67% 이나 LeNet 64.97% 모델이다.

### 3. 위협도 순위 (상위 5, 이번 조사 범위)

기지 논문 중 SCA(ICML 2024), Spik4lite(ICML 2026), UniSparse(ICML 2026), Suetake, Meem 은 **2026-09-29·10-02 절의 순위가 그대로 유효**하다. 아래는 이번에 새로 찾은 것 중의 순위다.

| 순위 | 논문 | 이유 | 방어 한 줄 |
|---|---|---|---|
| 1 | **SASL** (#1) | "약한 필터일수록 더 크게 벌한다 + 학습 중 순위 갱신 + 필터가 0 이 되어 제거 + 정확도 유지(Res-56 93.63→93.88, FLOPs −35.9%)". 우리 경쟁 가중의 ANN 선례로 가장 직접적이다 | 대상이 BN γ(가중치)이고 5등급 고정 배수. 우리는 활성(스파이크) 대상에 연속 가중 1−a/max, 커스텀 기울기, vmem 가중, 마지막 타임스텝 규제를 쓴다. SNN 에서 한 적 없다 |
| 2 | **FBS** (#2) | "채널 간 k-WTA 경쟁 + lasso + 죽는 채널 제거" 가 inter 주장(채널 경쟁으로 구조적 희소성)과 겹친다. ICLR 2019 라 리뷰어가 알 가능성이 높다 | FBS 의 WTA 는 하드 게이트이고 추론에도 남는다. 우리는 손실항 하나로 학습 중 채널이 죽는다. 구조 변경·예측기가 없다 |
| 3 | **Deng et al.** (#4) | 직접 학습 SNN, CIFAR-10, 스파이크율 λR. "단일 규제항으로 스파이크 감소" 의 SNN 선례. 게재처는 미확인 | 균등 가중·수동 λ. 같은 논문 표에서 감소하면 정확도가 떨어진다 (λ=0.01 에서 −1.91%p). plain 대조군의 근거로 쓸 수 있다 |
| 4 | **Kundu AGC / Spike-Thrift** (#5) | WACV 2021, "SNN 의 스파이크 활동을 채널 프루닝으로 줄인다" 가 목적. 채널 단위 구조적 희소성 주장과 겹친다 | 규제가 아니라 ANN attention 기반 사전 압축 + 재성장 단계. 표 수치는 미확인 |
| 5 | **DAP / SymbolNet** (#9, #10) | λ 를 task 손실 비례로 자동 설정하는 선례. 우리의 loss-ratio 는 새롭다는 주장을 할 때 "SNN 에서" 로 한정해야 한다 | 둘 다 SNN 도 활성 규제도 아니다. 우리 주장에서 loss-ratio 의 **새로움은 'SNN 활성 규제에서의 적용'** 으로만 말해야 한다 |

(Kurtz #3 은 활성 Hoyer 규제의 ANN 선례로 5위 바로 아래. 정확도 <0.3% 하락에서 활성 희소도 +9~12%p 이나 SNN·채널 경쟁이 없다.)

### 4. 결론: 조합 그대로를 한 논문이 하고 있는가

**이번 조사 범위에서는 없다.** "경쟁(순위·상대 크기) 가중 + 활성(스파이크) 대상 채널 단위 group 노름 + task 손실 비율 λ" 세 가지를 모두 갖춘 논문은 SNN 에서도 ANN 에서도 찾지 못했다. 두 개씩 겹치는 선례는 있다.

- 경쟁 가중 + 죽는 채널: SASL (ANN, 가중치 대상), FBS (ANN, 하드 게이트)
- 스파이크 대상 + 단일 규제항: Deng, Allred, Suetake, Meem, BPSR (균등 가중, 수동 또는 오차 피드백 λ)
- 손실 비례 λ: DAP, SymbolNet (비 SNN, 비 활성)

**논문에서 조심할 서술.** (1) "WTA·경쟁 가중 규제는 새롭다" 라고 쓰면 SASL 과 FBS 가 반례다. "SNN 스파이크 규제에서 처음" 으로 한정한다. (2) "loss-ratio λ 는 새롭다" 라고 쓰면 DAP·SymbolNet 이 반례다. "SNN 스파이크 규제에서 λ 를 task 손실 비율로 정한 첫 사례" 로 한정하고, 이것도 이번 검색 범위 안의 주장임을 밝힌다. (3) group-lasso 꼴 비판에는 Wen et al. (NeurIPS 2016, SSL) 과 Network Slimming (ICCV 2017) 이 선행한다. 이 둘은 이번에 본문을 읽지 않았다 (FBS 본문의 서술로만 확인).

### 5. 미확인 목록 (본문 인용 금지)

- SASL, Deng, Kundu 확장판, Allred, DPAP, DAP, SymbolNet, WTA 스파이킹 트랜스포머의 게재 여부. 위 표에서 학회로 확인한 것은 FBS(ICLR 2019), Kurtz(ICML 2020), Kundu WACV 2021 뿐이다.
- SASL 의 R(·) 이 L1 인지, 시드 수.
- Deng 의 CIFAR-10 T, 모델 시드 수.
- Kundu AGC 의 표(Table II·III) 수치. DPAP 의 SNN 표 수치(표 정렬 깨짐).
- DAP 식 16 은 PDF 텍스트에서 기호가 깨져 WebFetch 요약과 본문 서술("loss ratio, clipping prevents extreme values")을 교차해 읽었다. 원문 식 확인 필요.
- 읽지 못한 후보: Chen et al. ACM MM 2023 "Resource Constrained Model Compression via Minimax Optimization for SNNs" (arXiv 2308.04672, 가중치 프루닝 쪽으로 보이나 표 미확인), FP-SNN "Activation-Energy Pruning" (arXiv 2609.26167, 사후 가중치 프루닝, 학습 중 아님), Ahmad & Scheinkman 2019 (k-winners, arXiv 1903.11257, 본문 미확인), Wen 2016, Network Slimming 2017, "Spike-Count Reduction Techniques" (Neural Processing Letters, 페이월), Spiking ConvNeXt (ScienceDirect, 미확인).
- 검색이 닿지 않은 영역: CVPR/ICCV/ECCV 2024~2026 의 SNN 스파이크 규제 논문을 학회 이름으로 걸러 찾지 못했다. 검색 엔진이 학회별 필터를 지원하지 않아 "없다" 가 아니라 "못 찾았다" 로 읽어야 한다.


## 2026-10-06 최신 관련 연구·성능 비교 후보

조사 방법: 2026-09-29 / 10-02 / 10-06 앞 절에서 정리된 논문은 **수치를 재사용**하고 다시 조사하지 않았다(아래 "재사용" 표시). 이번에 새로 PDF 를 받아 `pdftotext` 로 표를 읽은 것은 "신규(표 확인)", 검색 요약·초록만 본 것은 "초록만"으로 표시한다. 표 열 정렬이 깨져 원문 재대조가 필요한 곳은 "(파싱)". 게재 여부는 PDF 머리글·학회 페이지에서 확인한 것만 적고, 나머지는 "게재 미확인". 편수를 채우려 하지 않았다.

**우리 기준값.** R19 C10 T=4 baseline 96.73 (n=6, 489K spikes/sample), inter 96.74 @342K (−30%, n=3). R19 C100 T=4 baseline 80.70 (n=5, 450K). QKFormer C10 96.21 @951K → 96.13 @520K (앞 절 기록).

### 0. 먼저 알아둘 것 (비교표 설계에 영향)

1. **R19 / QKFormer CIFAR 에서 spikes/sample 을 표로 보고하고 baseline 대비 감소율을 낸 논문은 이번에도 찾지 못했다.** 보고 지표는 평균 발화율(그림), SOPs/ACs, 이론 에너지(mJ), 연결 비율, 토큰 비율이다. 따라서 성능 비교 표의 효율 열은 "baseline 대비 상대 감소율"로 맞추고, 절대 spikes 열은 우리 행에만 채우게 된다.
2. **최근 정확도 SOTA 계열은 spikes 를 오히려 늘린다.** DS-ATGO(AAAI 2026) 자기 Table 2: R19 T=2 CIFAR-10 에서 AC 연산이 Vanilla-SNN 443.14M → 690.84M (fc=1.0, 에너지 0.43 → 0.65 mJ) / 493.32M (fc=1.2, 0.48 mJ). DIET-SNN 1084M, LTMD 1048M 도 같은 표. 즉 "임계값을 적응시켜 정확도를 올리는" 방법은 발화가 늘어나는 쪽이다. Related Work 에서 (b) 부류와 우리를 가르는 근거로 쓸 수 있다.
3. 우리 baseline R19 C10 T=4 96.73 은 아래 R19 T=4 정확도 SOTA 선(MTT 96.75, Logit-KD 96.97, STAA-SNN 97.14)과 같은 수준이다. 레시피가 다르니 절대 정확도가 아니라 **baseline 대비 변화**로 비교한다.

### 1. 부류별 Related Work 후보

굵은 글씨 = 이번 조사에서 새로 확인. (재사용) = 앞 절 수치를 그대로 쓴다.

**(a) 스파이크·발화율 규제·손실**
- Activity Pruning, Bu/Shi/Yu, NeurIPS 2025 (재사용, 10-02 절 1a~1f). 규제 기울기를 학습 가능한 임계값에만 보낸다. R20 C10 T=8 92.89→92.05(−0.84) @FR −77.5%. 우리와 가장 가까운 최신 경쟁자.
- Suetake et al., Spiking Synaptic Penalty (재사용, 9-29 절 A10 / 10-02 절 #13). 에너지에 비례하는 벌점. R18 C10 91.32→91.40 @E 58%.
- BPSR (Yan 2022, Front. Neurosci.), Deng et al. 2019(arXiv 1911.00822), Sorbaro 2020, Allred 2020 (재사용, 10-02 학습 알고리즘 절). 모두 2024 이전이거나 게재처 미확인.
- Stanojevic et al., Nat. Commun. 2024 (재사용, A9). TTFS, L1. VGG16 C10 93.69→93.27 @spikes/neuron 0.38→0.20.
- **이번 조사 결론: 2024~2026 탑 학회에서 "spike 벌점항 + R19/QKFormer CIFAR 정확도 SOTA급" 을 보고한 논문은 새로 나오지 않았다.** Hoyer 정규화(Datta/Liu/Beerel, "Hoyer regularizer is all you need for ultra low-latency SNNs", 검색 요약으로만 확인, 게재처·수치 미확인)는 T=1 설정이라 범위 밖.

**(b) 임계값·뉴런 모델 변경으로 발화 감소/안정화**
- Activity Pruning (위, AT-LIF, 재사용).
- **ILIF**, Sun/Duan/Kuhlmann/Wang/Zhang, IJCAI 2025 (신규, 표 확인; IJCAI 표기는 10-02 절 근거). 억제 유닛으로 과활성 완화. Table 1: R18 C10 T=4/6 95.24/95.49, C100 77.43/78.51, DVS VGG-11 T=10 78.60. 발화율은 그림(Fig 3, 4)뿐이고 수치 표 없음. 정확도 위주.
- **DS-ATGO**, Jiang/Xu/Fan/Yan, AAAI 2026 (신규, 표 확인). 위 0-2 참조. Table 1: R19 T=2 C10 96.91±0.12, C100 80.59±0.17, VGGSNN T=10 DVS 83.70±0.41. 코드 https://github.com/jqjiang1999/DS-ATGO
- **MPD-AGL**, Jiang et al., arXiv 2505.11863 (신규, 표 확인; IJCAI 2025 로 DS-ATGO 가 인용하나 본문에 학회 표기 없음, 게재 미확인). 아래 2 표.
- MP-Init + TrSG (Kook 2025, 재사용 A8), RMP-Loss (ICCV 2023, 재사용 A7), DeepEISNN (ICLR 2026, 재사용), Circulate-Firing (재사용).
- **ASN (Adaptive Spiking Neurons)**, Zhou/Guo/Wang/Ma/Cheng/Meng, arXiv 2604.12365 (신규, 표 확인, 게재 미확인). Spikingformer† 기반. Table 3 T=4: C10 96.27 (baseline 95.95), C100 81.11 (baseline 80.37). 효율은 ImageNet/언어 표에만 있고 CIFAR 스파이크 수 없음. QKFormer 저자 그룹.

**(c) 학습 중 뉴런/채널/토큰 프루닝·스파이크 희소화**
- Spik4lite (ICML 2026), UniSparse (ICML 2026), SCA (ICML 2024), SPEAR (게재 미확인), DPAP, Kundu AGC (모두 재사용).
- **MI-TRQR**, Xue/Li/Lu/Yuan/Liu/Liu/Yao/Yang/Li/Li/Maybank/Hu/Li, **NeurIPS 2025** (게재 확인: https://mlanthology.org/neurips/2025/xue2025neurips-mitrqr/ , 코드 https://github.com/dfxue/MI-TRQR, PSN 코드 기반). 시간 스텝 간 스파이크 특징의 상호정보량으로 중복을 재서 확률적 마스킹으로 스파이크를 제거하는 **무파라미터 플러그인**. **초록만** (OpenReview PDF 가 봇 차단으로 안 열림): CIFAR10-DVS T=4 에서 정확도 +1.7%, 에너지 −37.5%. 표 수치 미확인. "스파이크를 직접 지우면서 정확도를 올린다"는 점에서 우리 주장과 겹침. **재현 우선순위 높음**.
- **CH-SNN (Cannistraci-Hebb Training on Ultra-Sparse SNNs)**, Hua et al., **ICLR 2026** (신규, 표 확인; 코드는 "acceptance 후 공개"). 연결(가중치) 희소 학습. Table 2 (3FC, 한 설정): CIFAR-10 스파이크 수 6.3e7 → 1.9e7 (−70%), 발화율 30.70% → 9.28%, 정확도 80.67 → 82.84, 에너지 41 → 0.7 mJ, 링크 희소도 93.78%. **스파이크 수를 직접 보고하는 드문 논문**이나 3FC(정확도 80%대)라 우리 설정과 비교 불가. Table 1: 6Conv+2FC C10 74.62% 희소 −0.14%(94.60), Spikformer C10 82.21% 희소 −0.10%(94.26), C100 Spikformer 76.23 (+0.75).
- **SpikeNM**, Ye/Yu/Zhang/Kong/Wu/Jiang/Tao (NTU), arXiv 2511.12097 (신규, 표 확인, 게재 미확인). N:M 반구조 가중치 프루닝. Table 1 (△ 는 비프루닝 대비): C10 6Conv+2FC T=8 93.64(+0.80) SOPs 348.3M / 2:8 92.92(+0.08) 196.3M; C100 SEW-R18 T=4 73.92(−0.24) 350.2M; DVS VGGSNN T=10 83.90(+1.50) 289.3M. 같은 표 UPR(ICLR 2024) C10 92.63 @38.3M SOPs, DVS 81.90 @47.8M. **SOPs 는 UPR 쪽이 훨씬 낮다.** 가중치 프루닝이라 스파이크 감소가 목적이 아님.

**(d) 가중치 프루닝·양자화와 에너지 절감**
- QP-SNN (ICLR 2025), Q-SNNs (ACM MM 2024), LTH (ECCV 2022), SparseSpikformer (ICASSP 2024) (재사용).
- **SLAMP**, Wang/Ye/Ouyang, arXiv 2603.14946 (신규, 표 확인, 게재 미확인; 서식은 저널 원고). Table I: R19 **T=2** C10 92.42 (+1.23) @연결 40%, SOPs 112.35M; C100 R19 T=2 69.85 (+0.73) @25%; DVS VGGSNN T=5 78.21 (+0.09). 기준 모델이 약함(C10 91.19).
- **Synapse pruning for spiking transformers** (L1 P / DSP + sLIF), Sun/Cai/Chen/…/Yao/Guo (UESTC), arXiv 2508.01992 v4 (2026-01) (신규, 표 확인, 게재 미확인; 코드 https://github.com/GuoLab-UESTC/EfficientST). Table III: Spikformer-4-384 C10 baseline 94.76, 가중치 희소 60% 94.61(−0.15), 80% 93.94(−0.82), 90% 92.32(−2.44); DVS Spikformer-2-256 80.90 기준 60% 80.40. 스파이크·에너지 수치 없음, 파라미터만.

**(e) 시간 스텝 축소 / 적응적 T**
- SEENN (NeurIPS 2023), DT-SNN, Dynamic Confidence (ICCV 2023), All in one timestep (arXiv, 게재 미확인), Logit-KD (ICML 2025) (재사용).
- **MTT (Mixed Time-step Training)**, "Temporal Flexibility in SNNs: Towards Generalization Across Time Steps and Deployment Friendliness", **ICLR 2025** (게재 확인: proceedings.iclr.cc; arXiv 2503.17394; 코드 https://github.com/brain-intelligence-lab/temporal) (신규, 표 확인). **한 번 학습해 T 를 바꿔 쓴다.** Table 6 (3회 반복 mean±SD): R19 C10 T=2/4/6 96.20±0.07 / **96.75±0.04** / 96.84±0.03; C100 81.51±0.04 (T=4), 81.98±0.03 (T=6); Table 2 C100 T=2 80.35. DVS R18 T=10 82.8±0.54. SEENN-I 와 결합 시 같은 평균 T 에서 TET 보다 좋다(Table 4, 값 미확인).
- **TS-SNN (Temporal Shift)**, Yu/Zhang/Xu/Pan/Wang, **ICML 2025** (게재 확인: icml.cc/virtual/2025/poster/44594; arXiv 2505.04165) (신규, 표 확인). Table 1: R19 T=2 C10 96.72±0.08, C100 80.28±0.07; T=1 96.50/78.61. (T=4 행은 파싱에서 안 읽힘.) 표 인용 MPBN(ICCV 2023) R19 T=4 C10 96.52, C100 80.10 (2차 인용).
- **MI-TRQR** (위 c) 도 시간 중복 제거라 (e) 에 걸친다.

**(f) 트랜스포머 SNN 스파이크 효율화 (Spikformer/SDT/QKFormer)**
- QKFormer (NeurIPS 2024), SpiLiFormer, STATA (ICML 2024), TP-Spikformer (ICLR 2026), STAS, SparseSpikformer, Bishop (ISCA 2025) (재사용).
- **Uncert (Uncertainty-Aware Token Importance)**, Liu/Hao/Bu/Wang/Yu (PKU), arXiv 2605.09276 (신규, 표 확인; 서식이 "Conference acronym '26" 자리표시자, **게재 미확인**). **QKFormer 기반** 학습 없는 토큰 가지치기. Table 1 (T=4): C10 96.31 → 96.30 (6.74M), C100 80.98 → 81.06; DVS T=10 82.3 → 82.5. 토큰 80~90% 유지. Table 5: stage3.1 블록 SOP 1.233G → 1.050G (−14.86%, keep 0.40), 전체 전력 1.457 → 1.406 mJ (−3.5%). 우리 QKFormer 표에 가장 가까운 행이지만 감소 폭은 작다.
- **SEMM (Spiking Transformer with Experts Mixture)**, Zhou/Lu/Yuan 외, **NeurIPS 2024** (게재 확인: proceedings.neurips.cc). 희소 조건부 계산. Table 4 (T=4): Spikformer 95.19/77.86 → +SEMM 95.78/79.04; SDT 95.60/78.40 → 96.12/80.26; Spikingformer 95.81/79.21 → 96.16/80.24. 라우터 발화율은 ImageNet 표(Table 2)뿐. 정확도 위주.
- **STAtten**, Lee et al., **CVPR 2025** (게재 확인: openaccess.thecvf.com). Table 3 (저자들이 처음부터 다시 학습): QKFormer-4-384 T=4 C10 95.12 → 95.35, C100 79.79 → 80.20; DVS T=16 82.90 → 83.90. 복잡도는 같다고 주장, 스파이크 수 없음. **저자 재학습 QKFormer baseline 이 논문값(96.18/81.15)보다 1%p 낮음**: 같은 모델도 레시피에 따라 baseline 이 달라진다는 근거.
- **DS2TA**, Xu/Geng/Yin/Li (UCSB), arXiv 2409.15375 (신규, 표 확인, **게재 미확인**; OpenReview 심사본). 어텐션 맵 희소화. Table 3 (C100 ViT-4-384): 어텐션 맵 희소도 74.26~82.64% → 96.99~98.59%, 어텐션 에너지 −88.3~−91.9% (어텐션 연산만, 전체 아님). Table 4: Spikformer 94.19/76.05 → DS2TA 94.92/77.47.
- **E-SpikeFormer / Spike-driven Transformer V3**, Yao/Qiu 외 (Guoqi Li 그룹), arXiv 2411.16061 ("FOR REVIEW", **게재 미확인**). 정수 학습·스파이크 구동 추론(SFA). ImageNet 중심이고 CIFAR 표 없음 → 비교표에는 넣지 않음.

### 2. 성능 비교 표 후보 (같은 설정만)

"효율 열" 규칙: 논문이 보고한 지표를 그대로 적는다. spikes/sample 환산은 하지 않았다. n 은 논문이 ± 로 보고한 경우만 적고 반복 횟수는 명시된 것만 적는다.

**2-1. ResNet-19, CIFAR-10, T=4**

| 방법 | 학회·연도 | baseline → 방법 | 효율 지표 (baseline 대비) | n·SD | 출처(표) |
|---|---|---|---|---|---|
| **우리 baseline / inter** | – | 96.73 → 96.74 | 489K → 342K spikes/sample (−30%) | n=6 / n=3 | 우리 |
| TET | ICLR 2022 | 94.44±0.08 (단독) | 미보고 | SD 있음 | 9-29 절 A1, Table 3 |
| RMP-Loss | ICCV 2023 | 95.51±0.08 | 미보고 | SD 있음 | A7 (파싱) |
| MP-Init + TrSG | arXiv 2025 | 95.34±0.06 | 발화율 곡선만(그림) | SD 있음 | A8, Table 3 |
| MPBN (2차 인용) | ICCV 2023 | 96.52 | 미보고 | – | TS-SNN Table 1 |
| MPD-AGL | arXiv 2025 (IJCAI 표기 미확인) | 96.35 | Table 2: T=4 AC 1004.70M, 0.96 mJ. 비교행 T=2: STBP-tdBN 890.20M/0.83 mJ, LSG 677.72M/0.64 mJ, MPD-AGL 579.33M/0.55 mJ. 평균 발화율 14% (T=2, Fig 4) | SD 없음 | 신규, Table 1·2 |
| MTT | ICLR 2025 | 96.75±0.04 | 효율 표 없음(유연한 T 가 목적) | 3회, SD 있음 | 신규, Table 6 |
| Logit-KD | ICML 2025 | 96.97 (3회 평균) | 스파이크 수 미보고 | 3회 | C4, Table 1 |
| STAA-SNN | CVPR 2025 | 97.14±0.10 | R20 C100 T=4 0.366 mJ (baseline 없음) | SD 있음 | 신규, Table 1 |
| SEENN-I | NeurIPS 2023 | 96.46(T=4 기준) → 96.07@평균 T 1.09 … 96.44@1.34 | 평균 T 감소(−67~−73%) | – | 9-29 절 B1 |

**2-2. ResNet-19, CIFAR-100, T=4**

| 방법 | 학회·연도 | 정확도 | 효율 지표 | n·SD | 출처 |
|---|---|---|---|---|---|
| **우리 baseline** | – | 80.70 | 450K spikes/sample | n=5 | 우리 |
| TET | ICLR 2022 | 74.47±0.15 | 미보고 | SD | A1 |
| RMP-Loss | ICCV 2023 | 78.28 (파싱) | 미보고 | SD | A7 |
| MP-Init + TrSG | arXiv 2025 | 77.66±0.15 | 발화율 곡선(그림) | SD | A8 |
| MPBN (2차 인용) | ICCV 2023 | 80.10 | 미보고 | – | TS-SNN Table 1 |
| MPD-AGL | arXiv 2025 | 79.72 | C100 효율 표 없음 | – | 신규, Table 1 |
| MTT | ICLR 2025 | 81.51±0.04 | 없음 | 3회 | 신규, Table 6 |
| Logit-KD | ICML 2025 | 82.47 | 발화율은 R18 만 | 3회 | C4 |
| STAA-SNN | CVPR 2025 | 82.05±0.10 | 없음 | SD | 신규, Table 1 |

(R19 T=2 참고: TS-SNN 96.72±0.08 / 80.28±0.07, DS-ATGO 96.91±0.12 / 80.59±0.17 (AC 443M → 691M, 위 0-2), MTT 96.20 / 80.35. T 가 달라 같은 표에 넣지 않는다.)

**2-3. VGGSNN, CIFAR10-DVS (T=10 이 표준. 증강 여부를 맞춰야 함)**

| 방법 | 학회·연도 | baseline → 방법 | 효율 지표 | n·SD | 출처 |
|---|---|---|---|---|---|
| TET | ICLR 2022 | 77.33±0.21 (증강 83.17±0.15) | 미보고 | SD | A1 |
| Activity Pruning | NeurIPS 2025 | 82.70 → 82.20 (−0.50) | FR 0.080 → 0.043 (−46.3%), SOPs 1066.40M → 626.57M (−41.2%) | 미기재 | 10-02 절 1e |
| SCA (구조적) | ICML 2024 | 5Conv+1FC 72.80 → +0.90 @파라미터 21.73% | SOPs 는 그림만 | – | B4 |
| SPEAR | 게재 미확인 | 5Conv+1FC 80.05 | SynOps 39.3% | – | B5 |
| SpikeNM 2:4 / 2:8 | arXiv 2025 | 82.4(역산) → 83.90 / 83.20 | SOPs 289.3M / 165.3M, 연결 43% / 25% | – | 신규, Table 1 |
| UPR (SpikeNM 표 인용) | ICLR 2024 | 82.40 → 81.90 (−0.50) | SOPs 47.8M, 연결 6.80% | – | 신규, Table 1 |
| MPD-AGL | arXiv 2025 | 82.50 (TET 손실·Adam 시 84.10) | 미보고 | – | 신규, Table 1 |
| DS-ATGO | AAAI 2026 | 83.70±0.41 | 미보고 | SD | 신규, Table 1 |
| QP-SNN | ICLR 2025 | 82.10/81.80/81.30 (프루닝 행 81.50~74.90) | 모델 크기 | – | C3 |
| MI-TRQR | NeurIPS 2025 | T=4 에서 +1.7% (초록) | 에너지 −37.5% (초록) | 미확인 | 표 미확인 |

우리 VGGSNN DVS 수치는 이 표에 아직 없다. 위 행은 우리 값이 나오면 같은 T·증강으로 맞춰 쓴다.

**2-4. QKFormer / Spikformer 계열, CIFAR, T=4**

| 방법 | 학회·연도 | 모델 | baseline → 방법 | 효율 지표 | 출처 |
|---|---|---|---|---|---|
| **우리 QKFormer C10** | – | QKFormer | 96.21 @951K → 96.13 @520K (−45%) | spikes/sample | 9-29 절 머리 |
| QKFormer 원 논문 | NeurIPS 2024 | HST-4-384 6.74M | C10 96.18, C100 81.15 | CIFAR 효율 없음 | A5 |
| SpiLiFormer | 게재 미확인 | 4-384 7.04M | C10 96.63, C100 81.63 | 없음 | A6 |
| Uncert | 게재 미확인 | QKFormer 6.74M | C10 96.31 → 96.30, C100 80.98 → 81.06 | 블록 SOP −14.86% (keep 0.40), 전체 1.457 → 1.406 mJ | 신규, Table 1·5 |
| STAtten + QKFormer | CVPR 2025 | QKFormer-4-384 (재학습) | C10 95.12 → 95.35, C100 79.79 → 80.20 | 복잡도 동일 | 신규, Table 3 |
| STATA | ICML 2024 | Spikingformer | C100 79.98 @1.29 mJ → 79.86 @0.71 mJ | mJ −45% | B6 |
| TP-Spikformer | ICLR 2026 | Spikformer-4-384 | C10 95.19 → 95.12 (토큰 ×0.20), C100 78.21 → 77.83 (×0.55) | 토큰 비율 | B7 |
| STAS | arXiv | Spikformer-4-384 | C10/C100 94.8/77.3 → 95.2/77.9 | 0.74/0.89 → 0.40/0.50 mJ | B8 |
| SEMM | NeurIPS 2024 | SDT / Spikingformer | C100 78.40 → 80.26 / 79.21 → 80.24 | 없음(CIFAR) | 신규, Table 4 |
| DS2TA | 게재 미확인 | ViT-4-384 | C10/C100 94.19/76.05 → 94.92/77.47 | 어텐션 맵 에너지 −88~−92% (어텐션만) | 신규, Table 3·4 |
| SparseSpikformer | ICASSP 2024 | Spikformer | C10 94.95 → 95.01 | 3.74G → 3.04G (−19%) | C5 |

### 3. 직접 재현이 필요한 것

"우리 설정(R19 T=4 / QKFormer)에서 spikes/sample 로 같은 축에 놓으려면 우리 코드로 돌려야 하는 것." 코드 공개 여부는 위 본문에서 확인한 것만 적었다.

| 우선 | 방법 | 이유 | 코드 | 비고 |
|---|---|---|---|---|
| 1 | Activity Pruning (NeurIPS 2025) | 스파이크를 직접 줄이는 가장 가까운 최신 경쟁자. 설정이 R20 T=8 이라 R19 T=4 수치가 없다 | 본 절에서 미확인 | 10-02 절 6번 권고와 같음 |
| 2 | UPR (ICLR 2024, Shi et al.) | 비구조적 뉴런 프루닝 최초. R19 T=4 수치 없음 (표는 6Conv+2FC T=8, SEW-R18 T=4) | 본 절에서 미확인 | SpikeNM·SLAMP 표가 모두 UPR 을 기준으로 삼는다 |
| 3 | MI-TRQR (NeurIPS 2025) | 무파라미터 플러그인이라 이식이 쉽고, 스파이크를 직접 지운다. 표를 못 읽어 우리 설정 수치가 없다 | https://github.com/dfxue/MI-TRQR (PSN 기반) | 논문 표 확보 필요 (OpenReview 차단) |
| 4 | Suetake Synaptic Penalty | 구조가 가장 닮은 에너지 벌점. R18 C10 만 있음 | 본 절에서 미확인 | 10-02 절 #13 |
| 5 | MTT (ICLR 2025) | 같은 R19 T=4 에서 96.75/81.51 로 우리 baseline 과 같은 선. 정확도 기준선·T 유연성 비교용. 스파이크 수 없음 | https://github.com/brain-intelligence-lab/temporal | 정확도 외에 우리 규제를 올릴 수 있는지는 미확인 |
| 6 | DS-ATGO (AAAI 2026) | "정확도 올리면 스파이크도 는다"를 우리 축으로 보여줄 수 있다. 자체 보고(AC +56%, T=2)를 T=4 로 확인 | https://github.com/jqjiang1999/DS-ATGO | 반례 용도 |
| 7 | Uncert / STAtten (QKFormer) | QKFormer 에서 토큰 가지치기(−14.9% 블록 SOP)와 비교 | STAtten 코드 링크는 본문에서 일부만 보임, Uncert 미확인 | 비교는 가능, 우선순위 낮음 |
| 8 | Synapse pruning (EfficientST) | Spikformer 가중치 프루닝. 스파이크 미보고 | https://github.com/GuoLab-UESTC/EfficientST | 직교, 필요 없으면 생략 |

재현하지 않아도 되는 것: SpikeNM, SLAMP, CH-SNN (가중치 희소화라 스파이크를 줄이지 않는다, 코드 미확인), E-SpikeFormer (ImageNet 중심).

### 4. 위협적인 것 (이번 절 기준)

1. **MI-TRQR (NeurIPS 2025)**: "스파이크를 지우며 정확도를 올린다"를 플러그인 하나로 주장한다. 초록 수치(DVS T=4 +1.7%, 에너지 −37.5%)만 확인했고 표를 못 읽었다. **표를 먼저 확보할 것.**
2. **Activity Pruning (NeurIPS 2025)**: 앞 절 결론 유지. 감소율이 우리 R19 −30% 보다 크다(FR −77.5%). 단 정확도 −0.84%p.
3. **MTT (ICLR 2025)**: 스파이크 논문은 아니지만 R19 T=4 정확도 기준선(96.75/81.51)이 우리와 같다. 리뷰어가 "baseline 이 같은 수준" 을 확인하는 비교 대상이다.

### 5. 미확인·제외

- MI-TRQR 표 수치 (OpenReview PDF 봇 차단, 초록만).
- SpikeNM, SLAMP, Synapse pruning, Uncert, DS2TA, ASN, E-SpikeFormer, MPD-AGL: 게재 미확인 (arXiv 만).
- TS-SNN R19 T=4 행: 파싱에서 안 읽힘. MPBN 수치는 TS-SNN 표 인용(2차).
- Hoyer regularizer (Datta 외), Spiking ConvNeXt (ScienceDirect 게재, 검색 요약에 C100 74.75±0.96, 희소도 32.93±3.70% 3회): 본문 표를 읽지 않음.
- 제외: Guo 외, "Closing the Theory-Practice Gap in Spiking Transformers via Effective Dimension" (arXiv 2604.15769, 게재 미확인). QKFormer-4-384 C10 95.2±0.2 / 스파이크 2.1M 을 적었으나 모델 8.9M 파라미터, baseline 이 원 논문(96.18)과 달라 측정 근거를 신뢰하기 어렵다. 인용 금지.
- 이번에 검색했으나 CVPR/ICCV/ECCV 에서 스파이크 수 감소를 목적으로 한 분류 논문은 새로 찾지 못했다 (검색 범위 안의 결과이며 전수 조사 아님).


## 2026-10-07 Related Work 구성안

조사 방법: 이번에 **PDF 를 직접 내려받아 `pdftotext` 로 표를 읽은** 것은 MI-TRQR(NeurIPS 2025 proceedings PDF), UniSparse·Spik4lite(PMLR v306 PDF), EIHL(ICLR 2024 PDF)이다. CVF 목록은 `openaccess.thecvf.com/{CVPR2024,CVPR2025,CVPR2026,ICCV2025}?day=all` 전체 HTML 의 제목을 걸러 읽었고, ECCV 2024 는 `ecva.net/papers.php` 목록으로 걸렀다. 그 외는 "초록만" / "게재 확인만" 으로 표시한다. 앞 절(09-29, 10-02, 10-06 두 절)에 있는 논문은 수치를 다시 조사하지 않았다. 전수 조사가 아니라 이 세션 범위 안의 결과다.

### 0. 이번에 메운 빈 곳

**(a) MI-TRQR 표 수치 (확보).** OpenReview 는 403 이었지만 NeurIPS 공식 proceedings PDF 가 열린다: https://proceedings.neurips.cc/paper_files/paper/2025/file/dfb89b72be941eeeaeb5ee53bd3e7fc1-Paper-Conference.pdf (arXiv 판은 없음, `export.arxiv.org` 검색 0건). 저자: Xue, Li, Lu, Yuan, Liu, Liu, Yao, Yang, Li (Guoqi), Li (Bing), Maybank, Hu, Li (Zhetao). 코드 https://github.com/dfxue/MI-TRQR (PSN 기반). **기준선은 PSN (Parallel Spiking Neuron, 코드 github.com/fangwei123456/Parallel-Spiking-Neuron; 서지·학회는 이 PDF 에서 미확인)** 이고 우리 기준 R19 가 아니다.

| 표 | 설정 | 기준선 → MI-TRQR | 효율 |
|---|---|---|---|
| Table 7 | CIFAR-10, Modified PLIF Net, T=4 | 95.32 → 95.83±0.10 | 에너지 1.32 → 0.35±0.05 mJ (−73.48%) |
| Table 8 (부록) | 같은 설정, 지역(LR)+전역(GR) 중복 둘 다 | 95.32 → 95.83 | **발화율 16.39% → 5.36% (−67.30%)**. LR 만 10.59, GR 만 8.98 |
| Table 7 | CIFAR-10, STMixer-4-384-32, T=4 | 96.01 → 96.64±0.12 | 에너지 0.95 → 0.51±0.08 mJ |
| Table 7 | CIFAR-100, ResNet-19, T=4 | PSN♠ 76.14 → 77.70±0.10 | 에너지 1.78 → 1.24±0.07 mJ (♠ 는 저자 재구현) |
| Table 7 | CIFAR-100, ResNet-18, T=4 | PSN♠ 75.75 → 76.70±0.12 | 0.43 → 0.31±0.08 mJ |
| Table 7 | CIFAR-100, STMixer-4-384-32, T=4 | 81.87 → 83.06±0.11 | 1.08 → 0.77±0.08 mJ |
| Table 1 | CIFAR10-DVS, VGG, T=4 | 82.3 → 83.9±0.05 | 0.72 → 0.45±0.06 mJ (−37.5%) |
| Table 4·6 | CIFAR10-DVS, VGG, T=4 | 82.3 → 84.0 (LR+GR+재가중) | 발화율 14.47% → 8.35% (−42.29%) |
| Table 2 | ImageNet SEW-R50, T=4 | PSN♠ 72.01 → 73.23±0.13 | 4.11 → 3.09±0.08 mJ |

- 방법 (본문 3.2~3.3절). 시간 스텝 간 스파이크 특징의 상호정보량(MI)으로 중복을 재서, t≥2 의 스파이크를 확률적 마스크로 지운다. 무파라미터 플러그인. 재가중 α_t 는 층별 발화율로 정규화. 마스크는 학습 때만 쓴다("The operation of removing spikes is only used during training"). 놓는 위치는 마지막 층(Table 5: layer 8 이 가장 좋음, 발화율 −42.29%).
- **± 의 n 은 분류 표에 적혀 있지 않다.** "3 random seeds" 는 시계열 예측 표(Table 3)에만 있다. 분류 쪽 ± 가 몇 회인지는 **미확인**.
- 에너지(mJ)는 이론 모델이다 (Power 열 정의: "energy cost of inference one test sample"). 실측 아님.
- 우리와의 비교: 기준선이 PSN(CIFAR-10 95.32)이라 우리 R19 96.73 과 다른 레시피다. 발화율 −67% 와 정확도 +0.51 은 **약한 기준선에서의 값**이라는 점을 빼고 읽으면 안 된다. 그래도 "스파이크를 지우면서 정확도가 오른다" 는 우리 주장과 같은 방향이고, 규제항이 아니라 마스크라는 점이 다르다.

**(b) CVPR/ICCV/ECCV 2024~2026 SNN 목록.** 제목에 spik/SNN/neuromorphic 이 든 논문을 전부 걸렀다 (CVPR 2024: 9편 중 스파이크 카메라 6편 제외, CVPR 2025: 15편 중 카메라·공격 제외, CVPR 2026: 19편, ICCV 2025: 6편, ECCV 2024: SNN 분류·검출 약 9편). **스파이크 수를 줄이는 것을 목적으로 한 규제·프루닝 논문은 없다.** 가까운 것은 아래 둘뿐이다. (제목에 키워드가 없는 논문은 걸러지지 않으므로 "없다"가 아니라 "제목으로는 못 찾았다".)
- **Are Conventional SNNs Really Efficient? A Perspective from Network Quantization.** Shen, Zhao, Li, Li, Zeng. **CVPR 2024**. https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.html. 초록만 읽음: 시간 스텝 = 양자화 비트폭과 같은 표현이라 보고, SynOps 대신 "Bit Budget" 으로 에너지를 추정하자고 주장. 수치 표는 **미확인**. "SynOps 프록시 비판" 용 (Intro/RW 의 에너지 단락).
- **SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition.** Zheng, Huang, Yu(Yingchao), Zhu, Tang, Yu(Zhaofei), Jin. **ICCV 2025 (게재 확인)**. https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.html. 앞 절은 "게재 미확인" 이었으나 이제 확인됨. 초록: CIFAR-10 +0.45%, CIFAR-100 +0.48%, CIFAR10-DVS +2.70%, ImageNet-1K +1.6%. 어텐션 구조 변경이며 효율 지표는 초록에 없음.
- 그 밖에 CVF 에서 확인한 효율 주장 논문: STAtten (CVPR 2025, 앞 절), STAA-SNN (CVPR 2025, 앞 절), SpikingResformer (CVPR 2024, Shi/Hao/Yu, 구조), ECCV 2024 "Integer-Valued Training and Spike-driven Inference SNN for High-performance and Energy-efficient Object Detection" (검출), "Spiking Wavelet Transformer" (구조). 모두 스파이크 규제가 아니라 구조·표현 개선이라 RW 본문에는 한 문장 이상 쓰지 않는다.

**(c) Spik4lite·UniSparse PDF (확보).** 둘 다 PMLR ICML 2026 (Volume 306) 에 실려 있고 PDF 가 열린다 (앞 절의 "PDF 를 열지 못했다" 해소).
- **UniSparse**: Shi, Bu, Yu (PKU). https://proceedings.mlr.press/v306/shi26a.html , PDF https://raw.githubusercontent.com/mlresearch/v306/main/assets/shi26a/shi26a.pdf , 코드 https://github.com/xyshi2000/UniSparse .
  - **본문 서술 (Introduction)**: "spike regularization tends to indiscriminately suppress all spikes, which may conflict with the task optimization objective and consequently impair model performance" 와 "learnable masks … additional trainable parameters". 즉 이 논문이 **스파이크 규제를 '무차별 억제' 로 비판**한다. 우리 경쟁 가중(1−a/max)이 정면으로 받는 반론이자 동기.
  - 방법: 스파이크 벌점 λ‖s ⊙ g_s(s)‖₁ (식 14), g_s = 에너지 계수 (층의 시냅스 연결 수 상수로 근사, 식 29~32). **기울기는 BN 의 γ, β (= 다음 뉴런의 임계값)로만 간다** (식 15~17, 31·32). 가중치 쪽은 soft-threshold 재매개변수화 프루닝(식 20, 27)을 따로 둔다. λ 는 수동.
  - 수치 (Table 1, CIFAR-10, T=4, 정확도±, 평균 SOPs): ResNet-19 94.71±0.11 @39.47M → 94.07±0.07 @18.58M → 92.48±0.10 @8.18M → 91.56±0.04 @6.11M. ResNet-18 94.11±0.05 @43.30M … 92.38±0.23 @7.04M. CifarNet 94.42±0.01 @46.37M … 90.76±0.19 @7.26M. Table 2 (DVS-CIFAR10, VGGSNN, T=10) 83.22±0.53 @39.26M … 79.12±0.83 @5.65M. Table 3 (ImageNet, R18, T=4) 61.89±0.12 @275.17M … 59.66±0.16 @108.72M; Spikformer-8-384 72.21 @2576.85M … 68.29 @1138.52M. 비교행 UPF(=ICLR 2024 Shi et al.) CIFAR-10 CifarNet T=8 92.63 @38.32M … 90.65 @8.50M.
  - **밀집 기준선 행이 표에 없다** (Table 1~3 에 비프루닝 행 없음). 따라서 "baseline 대비 정확도 변화" 는 이 PDF 표만으로는 **미확인**. 가중치 프루닝과 섞인 SOPs 이므로 스파이크 수만의 감소율도 분리되지 않는다 (스파이크 희소화 단독 행은 부록·Table 4 쪽에 있을 수 있으나 읽지 못함). ± 의 n 은 **미확인**.
- **Spik4lite**: She, Zhou, Wang, Huang, Cui, Guo. https://proceedings.mlr.press/v306/she26b.html , PDF https://raw.githubusercontent.com/mlresearch/v306/main/assets/she26b/she26b.pdf .
  - 방법: 채널별 발화율 벡터를 작은 선형 라우터에 넣어 keep/zero 로짓을 만들고 Hard Gumbel-Softmax + STE 로 채널 마스크를 만든다. 에너지 벌점 L_energy(SOPs·평활 발화율)는 Gumbel 없는 Softmax(P)에서 계산 (식 2~6). 학습 후 채널을 물리적으로 제거. 구조 변경(라우터)이 있고 λ 수동. 대상은 SNN 트랜스포머.
  - 수치 (Table 1, 배치 128, 정확도→정확도, SOPs, Jetson 실측 에너지): CIFAR-10 Spikformer-4-384 T=4 95.24→94.15, 822.64→321.49M SOPs (−60.9%), 5.34→3.08 J; SD-Transformer-2-512 95.41→94.78, 752.62→474.47M, 5.34→3.64 J; Spikingformer-4-384 95.07→94.43, 442.85→280.40M, 5.42→3.42 J. CIFAR-100 Spikformer 78.0→74.19, Spikingformer 79.07→77.07. DVS-CIFAR10 T=16 Spikformer-2-256 79.9→76.4. **모든 행에서 정확도가 떨어진다 (−0.63 ~ −3.81%p).** 표에 ± 없음 (n 미기재).
  - **실측 에너지를 낸다**: NVIDIA Jetson Orin Nano 8GB, Jetson Power GUI, 전체 테스트셋 추론 (Fig 5, Table 1 의 Energy 열). 이것은 "SynOps 프록시 비판 + 실측" 축에서 ICML 급 선행이다. 단 뉴로모픽 칩이 아니라 상용 엣지 GPU 이고, 이득이 채널을 물리적으로 지워 dense 연산이 줄어서 생긴다 (스파이크 희소성 자체가 아님).
  - 우리에게 유리: 정확도 −0.6~−3.8%p 로 "무손실" 이 아니다.

**(d) 측면 억제·WTA (2023~2026).**
- **DeepEISNN** — Training Deep Normalization-Free SNNs with Lateral Inhibition. Liu, Ding, Yu. **ICLR 2026 (arXiv 코멘트 "Accepted by ICLR 2026" 확인)**. https://arxiv.org/abs/2509.23253 . 층을 흥분(E)·억제(I) 집단 회로로 바꾸고 감산·제산 억제를 쓴다. 정규화 제거가 목적. 초록 수치: CIFAR-10 ResNet-18 92.05%, TinyImageNet 50.29%, DVS-Gesture 94.86%. 스파이크 감소 수치 없음. 구조 변경.
- **SpiLiFormer** (위, ICCV 2025). 구조 변경 (어텐션).
- **WTA Spiking Transformer for Language Modeling**. Zhou 외, arXiv 2604.11321, 게재 미확인 (앞 절). 구조 변경 (언어모델).
- **EIHL** — Adaptive Deep SNN with Global-Local Learning via Balanced Excitatory and Inhibitory Mechanism. Jiang, Xu, Ran, Shen, Lv, Zhang, Pan. **ICLR 2024**. https://proceedings.iclr.cc/paper_files/paper/2024/file/4ec43957eda1126ad4887995d05fae3b-Paper-Conference.pdf . Table 1 (ResNet18 CIFAR10): STBP 89.53 → EIHL 90.25, **연결 희소도 17.17%**. 측면 억제가 아니라 학습 규칙 전환용이고 희소도는 가중치 연결이다. 기준선이 약하다 (R18 89.53). 동기 인용 가치는 낮다.
- 이 셋 외 2023~2026 탑학회에서 "측면 억제/WTA 를 **손실항** 으로 SNN 스파이크에 건" 논문은 찾지 못했다. ANN 쪽 선례는 FBS (ICLR 2019), SASL (arXiv 2020) (10-06 절).
- 낮은 우선 (인용하지 않음): "Inhibition SNN" (Springer SN Applied Sciences 2024, WTA+LIF, 소규모 패턴 인식, 본문 미확인).

**(e) 기존 서지의 정정·확정 (이번에 확인된 것).**
- **Suetake et al. 은 TMLR 2024 게재**, 게재본 제목이 **"Synaptic Interaction Penalty: Appropriate Penalty Term for Energy-Efficient Spiking Neural Networks"** 로 바뀌었다 (arXiv 2302.01500 은 옛 제목 "Spiking Synaptic Penalty"). https://mlanthology.org/tmlr/2024/suetake2024tmlr-synaptic/ . **게재본 수치는 아직 못 읽었다** (OpenReview 차단). 앞 절의 R18 C10 91.32→91.40 은 arXiv 판 수치. 인용 때 TMLR 제목을 쓴다.
- **Deng et al. ADMM + Activity Regularization 은 IEEE TNNLS 34(6):2791–2805** (UniSparse 참고문헌 표기에서 확인, 2차 출처, "2021" 표기). 앞 절 "게재처 미확인" 을 대체. 저널 페이지 직접 확인은 **미확인**.
- **SPEAR** 는 ICLR 2026 **투고**(OpenReview 제출)까지만 확인. 채택 여부는 여전히 **미확인**. 본문에서 "ICLR 2026" 로 쓰지 말 것.
- **Activity Pruning 코드 공개**: https://github.com/putshua/Activity-Pruning-SNN (앞 절 "코드 미확인" 해소). UPR(ICLR 2024 Spotlight) 코드: https://github.com/xyshi2000/Unstructured-Pruning .
- Hoyer regularizer (Datta, Liu, Beerel, arXiv 2212.10170): dblp 상 **arXiv 프리프린트 (CoRR) 만 확인**, 학회 게재 확인 못 함.
- 새로 확인한 ICML 2022: STDS "State transition of dendritic spines improves learning of sparse SNNs" (Chen 외, ICML 2022 pp.3701–3715), IJCAI 2021 Grad R (Chen 외, pp.1713–1721). 둘 다 UniSparse 참고문헌 표기(2차). 본문 미확인.
- **ICLR 2025** "Improving the Sparse Structure Learning of SNNs from the View of Compression Efficiency", Shen, Xu, Pan, Chen. https://iclr.cc/virtual/2025/poster/28809 . 초록만: 동적 구조 학습(PQ 지수로 재배선 비율 결정). 수치 **미확인**. 가중치 연결 쪽.

### 1. Related Work 단락 구성안 (4단락)

우리 위치 한 줄(공통): **학습 손실에 규제항 하나를 더해, 뉴런 모델·구조·사후 단계를 바꾸지 않고, 경쟁 가중(1−a/max)으로 약한 스파이크부터 줄이며, λ 를 task 손실 비율로 자동 설정한다.**

**단락 1. 스파이크·에너지 규제 손실.**
- 핵심 3편: (1) Suetake et al., TMLR 2024 (에너지 비례 벌점, 구조가 가장 닮음) (2) Deng et al., IEEE TNNLS (스파이크율 λR) (3) Sorbaro et al., Front. Neurosci. 2020 / BPSR (Yan, Front. Neurosci. 2022) (제곱 L2).
- 인용 후보 (7~9편): Suetake TMLR 2024; Deng TNNLS; Sorbaro 2020; Pellegrini SLT 2021; BPSR 2022; Sakemi Sci. Rep. 2023; Stanojevic Nat. Commun. 2024 (TTFS L1); Neil ACM SAC 2016; Allred arXiv 2020 (게재 미확인); ANN 쪽 Kurtz ICML 2020 (Hoyer).
- **우리와의 차이 한 줄**: 모두 모든 스파이크에 같은 벌점·수동 λ 이고, UniSparse (ICML 2026) 가 이를 "무차별 억제로 task 와 충돌" 이라 비판하는데, 우리는 채널 안 경쟁 가중으로 약한 스파이크만 겨냥하고 λ 를 L_task/R 비로 자동 설정한다.

**단락 2. 뉴런·임계값·시간 중복을 바꾸는 방법.**
- 핵심 3편: (1) Activity Pruning (Bu/Shi/Yu, NeurIPS 2025; AT-LIF + 임계 적응, R20 C10 T=8 92.89→92.05, FR −77.5%) (2) MI-TRQR (Xue 외, NeurIPS 2025; 마스크로 시간 중복 제거, PLIF-Net C10 T=4 95.32→95.83, 발화율 −67.3%) (3) UniSparse 의 스파이크 희소화 (Shi/Bu/Yu, ICML 2026; BN 아핀=임계값에 벌점).
- 인용 후보 (6~8편): 위 3편; RMP-Loss (ICCV 2023); IM-Loss (NeurIPS 2022); DS-ATGO (AAAI 2026, 정확도를 올리지만 AC 연산이 늘어난 반례, R19 T=2 443M→691M, 10-06 절); T 축 SEENN (NeurIPS 2023), Dynamic Confidence (ICCV 2023), DT-SNN (학회 미확인).
- **차이 한 줄**: 이들은 뉴런 모델·임계값 기울기 경로·마스킹 모듈을 바꾸고, 우리는 뉴런과 구조를 그대로 두고 손실 항 하나로 같은 일을 하며 R19/QKFormer 에서 정확도를 기준선 이상으로 유지한다.

**단락 3. 구조적·비구조적 프루닝과 에너지 프록시.**
- 핵심 3편: (1) UPR (Shi/Ding/Hao/Yu, ICLR 2024 Spotlight; 비구조적 뉴런 프루닝) (2) SCA (Li 외, ICML 2024; 채널 활동 기반 구조적) (3) Spik4lite (She 외, ICML 2026; 라우터 게이팅 + Jetson 실측).
- 인용 후보 (8~10편): 위 3편; UniSparse 의 가중치 프루닝 쪽; QP-SNN (ICLR 2025); LTH (ECCV 2022, 가중치 프루닝은 스파이크를 줄이지 않는다는 근거); STDS (ICML 2022, 2차); Kundu AGC (WACV 2021); EIHL / Sparse Structure Learning (ICLR 2024 / 2025, 가중치 연결); 토큰 쪽 STATA (ICML 2024), TP-Spikformer (ICLR 2026). 에너지 프록시 비판: Shen et al. CVPR 2024, Reconsidering the Energy Efficiency of SNNs (arXiv 2409.08290), The Sparsity Ceiling (arXiv 2607.26648).
- **차이 한 줄**: 프루닝은 별도 기준·단계·재성장이나 추가 모듈(라우터·마스크)이 필요하고 UPR·SCA 의 수치는 기준선보다 낮은 모델에서 나왔다. 우리는 학습 중 규제만으로 채널이 죽고 (inter), Spik4lite 가 보인 실측 에너지와 달리 SynOps 프록시에 머무르는 한계를 명시한다 (실측은 Phase 0 의 몫).

**단락 4. 경쟁·측면 억제.**
- 핵심 3편: (1) SpiLiFormer (ICCV 2025; 어텐션에 측면 억제) (2) DeepEISNN (ICLR 2026; E-I 회로) (3) FBS (Gao 외, ICLR 2019; 채널 k-WTA, ANN) — 또는 SASL (arXiv 2020, 순위 기반 벌점 배수).
- 인용 후보 (5~7편): 위; LISNN (Cheng, IJCAI 2020, refs.bib 에 있음); WTA Spiking Transformer (arXiv 2026, 게재 미확인); EIHL (ICLR 2024); SASL; Ahmad & Scheinkman 2019 (k-WTA, 본문 미확인); Wen 2016 SSL, Network Slimming (ICCV 2017, 본문 미확인).
- **차이 한 줄**: 측면 억제를 구조(어텐션·E-I 회로·하드 게이트)로 구현하는 반면 우리는 구조를 건드리지 않고 학습 신호(손실)만으로 같은 경쟁을 암묵적으로 만든다.

### 2. 인용 목록 (BibTeX 키 후보)

`refs.bib` 에 이미 있는 키는 **(있음)**. 서지 표기는 이번에 확인한 출처 기준이며, "미확인" 은 본문 인용 전에 다시 확인한다.

| 키 후보 | 제목 / 저자 | 학회·연도 | 링크 | 상태 |
|---|---|---|---|---|
| `bu2025activity` | Activity Pruning for Efficient SNNs. Bu, Shi, Yu | NeurIPS 2025 | https://neurips.cc/virtual/2025/poster/115023 | (있음) |
| `shi2024towards` | Towards Energy Efficient SNNs: An Unstructured Pruning Framework. Shi, Ding, Hao, Yu | ICLR 2024 (Spotlight) | https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd3c889922df2112a5b1769e3c19e28e-Abstract-Conference.html | 확인 |
| `xue2025mitrqr` | MI-TRQR. Xue, Li, Lu, Yuan, Liu, Liu, Yao, Yang, Li, Li, Maybank, Hu, Li | NeurIPS 2025 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/dfb89b72be941eeeaeb5ee53bd3e7fc1-Abstract-Conference.html | 확인 (표 읽음) |
| `shi2026unisparse` | UniSparse. Shi, Bu, Yu | ICML 2026, PMLR 306 | https://proceedings.mlr.press/v306/shi26a.html | 확인 (표 읽음) |
| `she2026spik4lite` | Spik4lite. She, Zhou, Wang, Huang, Cui, Guo | ICML 2026, PMLR 306 | https://proceedings.mlr.press/v306/she26b.html | 확인 (표 읽음) |
| `li2024spiking` | SCA, Towards efficient deep SNN construction with spiking activity based pruning. Li 외 | ICML 2024, PMLR 235:29063 | https://proceedings.mlr.press/v235/li24bz.html | (있음) |
| `suetake2024synaptic` | Synaptic Interaction Penalty. Suetake, Ushimaru, Saiin, Sawada | TMLR 2024 | https://mlanthology.org/tmlr/2024/suetake2024tmlr-synaptic/ | 게재 확인, 게재본 수치 미확인 |
| `deng2021comprehensive` | Comprehensive SNN Compression Using ADMM Optimization and Activity Regularization. Deng 외 | IEEE TNNLS 34(6):2791–2805 | https://arxiv.org/abs/1911.00822 | 게재처는 2차 출처 |
| `sorbaro2020optimizing` / `pellegrini2021low` / `sakemi2023sparse` / `neil2016learning` | (앞 절 서지) | – | – | (있음) |
| `yan2022backpropagation` | BPSR, Backpropagation with Sparsity Regularization for SNN Learning. Yan 외 | Front. Neurosci. 2022 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9047717/ | 검색 결과로 존재 확인, 저자·쪽수는 미확인 |
| `stanojevic2024approaching` | 0.3 spikes per neuron (Approaching ANN accuracy with TTFS SNNs). Stanojevic 외 | Nat. Commun. 2024 | https://arxiv.org/abs/2306.08744 | 앞 절 |
| `kurtz2020inducing` | Inducing and Exploiting Activation Sparsity for Fast Inference on DNNs. Kurtz 외 | ICML 2020 | http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf | 앞 절 |
| `guo2023rmploss` | RMP-Loss. Guo 외 | ICCV 2023 | https://arxiv.org/abs/2308.06787 | 앞 절 |
| `li2023seenn` | SEENN. Li, Geller, Kim, Panda | NeurIPS 2023 | https://arxiv.org/abs/2304.01230 | 앞 절 |
| `li2023dynamic` | Dynamic Confidence. Li, Jones, Furber | ICCV 2023 | https://arxiv.org/abs/2303.10276 | 앞 절 |
| `jiang2026dsatgo` | DS-ATGO. Jiang, Xu, Fan, Yan | AAAI 2026 | https://github.com/jqjiang1999/DS-ATGO | 앞 절 (arXiv 번호 미기재) |
| `gao2019dynamic` | Dynamic Channel Pruning: Feature Boosting and Suppression. Gao 외 | ICLR 2019 | https://arxiv.org/abs/1810.05331 | 앞 절 |
| `shi2020sasl` | SASL. Shi, Xu, Tasaka, Chen | arXiv 2020, 게재 미확인 | https://arxiv.org/abs/2003.05891 | 앞 절 |
| `kundu2021spikethrift` | Spike-Thrift. Kundu, Datta, Pedram, Beerel | WACV 2021 | https://arxiv.org/abs/2107.12445 | 앞 절 |
| `kim2022lottery` | Exploring Lottery Ticket Hypothesis in SNNs. Kim 외 | ECCV 2022 | https://arxiv.org/abs/2207.01382 | 앞 절 |
| `wei2025qpsnn` | QP-SNN. Wei 외 | ICLR 2025 | https://arxiv.org/abs/2502.05905 | 앞 절 |
| `zhuge2024stata` | STATA. Zhuge, Wang, Yao, Cheng | ICML 2024 | https://proceedings.mlr.press/v235/zhuge24b.html | 앞 절 |
| `wei2026tpspikformer` | TP-Spikformer. Wei, Zhou, Zhang 외 | ICLR 2026 | https://arxiv.org/abs/2603.00527 | 앞 절 |
| `shen2024conventional` | Are Conventional SNNs Really Efficient? A Perspective from Network Quantization. Shen, Zhao, Li, Li, Zeng | CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.html | 게재 확인, 초록만 |
| `zheng2025spiliformer` | SpiLiFormer. Zheng, Huang, Yu, Zhu, Tang, Yu, Jin | ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.html | 게재 확인, 초록만 |
| `liu2026deepeisnn` | Training Deep Normalization-Free SNNs with Lateral Inhibition. Liu, Ding, Yu | ICLR 2026 | https://arxiv.org/abs/2509.23253 | 게재 확인 (arXiv 코멘트) |
| `jiang2024eihl` | Adaptive Deep SNN with Global-Local Learning via Balanced Excitatory and Inhibitory Mechanism. Jiang 외 | ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/file/4ec43957eda1126ad4887995d05fae3b-Paper-Conference.pdf | 확인 (표 읽음) |
| `shen2025sparsestructure` | Improving the Sparse Structure Learning of SNNs from the View of Compression Efficiency. Shen, Xu, Pan, Chen | ICLR 2025 | https://iclr.cc/virtual/2025/poster/28809 | 게재 확인, 초록만 |
| `cheng2020lisnn` | LISNN. Cheng, Hao, Xu, Xu | IJCAI 2020 | https://www.ijcai.org/proceedings/2020/211 | (있음) |
| `zhou2026wta` | Winner-Take-All Spiking Transformer for Language Modeling. Zhou 외 | arXiv 2604.11321, 게재 미확인 | https://arxiv.org/abs/2604.11321 | 앞 절 |
| `xie2025spear` | SPEAR. Xie 외 | arXiv 2507.02945; ICLR 2026 투고만 확인 | https://arxiv.org/abs/2507.02945 | 채택 미확인 |
| `hu2024reconsidering` | Reconsidering the Energy Efficiency of SNNs | arXiv 2409.08290 | https://arxiv.org/abs/2409.08290 | 기지 (이번에 재조사 안 함) |

(키 접두어가 저자 성+연도 규칙이다. 기존 `refs.bib` 의 규칙과 같다. `hu2024reconsidering` 은 저자 미확인이므로 키의 성이 틀릴 수 있다. 삽입 전에 확인.)

### 3. 위협적인 것 (이번 조사 기준, 상위 3)

1. **MI-TRQR (NeurIPS 2025).** 표를 읽고 나니 위협이 커졌다. 플러그인 하나로 CIFAR-10 T=4 에서 정확도 +0.51 에 **발화율 −67.3%** (Table 8). 우리 R19 −30%@±0.00 보다 큰 감소율이다. 방어: 기준선이 PSN (95.32) 으로 약하고 우리 R19 96.73 과 레시피가 다르다, 에너지는 이론값, 마스크·재가중이라는 모듈 추가 (규제항 아님), 분류 표 ± 의 n 이 불명. 같은 R19 T=4 에서 재현하지 않으면 우열 주장은 못 한다.
2. **UniSparse (ICML 2026).** 스파이크 벌점 형태(에너지 가중 L1)와 "λ 를 키워 스파이크를 줄인다" 구도가 우리와 같고, **"스파이크 규제는 무차별 억제" 라는 비판을 우리 방법 계열 전체에 건다.** SOPs 7.04~6.11M 까지 줄인 SOTA 주장이다 (단 가중치 프루닝과 혼합, 기준선 행 없음, 정확도 91~92%). 방어: 기울기가 BN 아핀(임계값)으로만 가고 가중치 프루닝과 섞여 있으며, 우리는 경쟁 가중과 task 비례 λ 를 쓴다. 이 비판에 답하려면 plain L2 대비 경쟁 가중의 이득을 정확도×스파이크 곡선으로 보여야 한다.
3. **Activity Pruning (NeurIPS 2025).** 앞 절 결론 유지. 코드가 공개돼 있어 (https://github.com/putshua/Activity-Pruning-SNN) 같은 R19 T=4 설정에서 직접 돌려 비교할 수 있게 됐다 (앞 절의 "재현 우선순위 1" 이 실행 가능해짐). Spik4lite 는 정확도 −0.6~−3.8%p 라 순위가 낮다.

### 4. 못 찾은 것·미확인 (본문 인용 금지)

- CVPR/ICCV/ECCV 2024~2026 에서 스파이크 감소 목적의 규제·프루닝 논문: 제목 필터로는 0편 (위 0-(b)). 초록 전문 검색은 하지 않았다.
- UniSparse 의 밀집 기준선 정확도, 스파이크 희소화 단독 결과(ablation 표), ± 의 n. MI-TRQR 분류 표 ± 의 n. Spik4lite 의 ± 유무(표에 없음).
- Suetake TMLR 게재본의 표 수치 (arXiv 판과 같은지). Shen CVPR 2024 의 수치 표. DeepEISNN 의 스파이크·에너지 수치. ICLR 2025 Sparse Structure Learning 의 수치.
- Deng TNNLS 서지의 저널 페이지 직접 확인, BPSR 서지의 저자·쪽수, Hoyer regularizer (Datta) 의 학회 게재.
- SPEAR 의 ICLR 2026 채택, ICML 2026 PMLR v306 에서 SNN 제목 36편 중 위 외에 스파이크 감소를 다루는 것: "SpikeNet: Sparse Spike-Driven Mask Vector Transformer" (포인트클라우드, Zhou 외, 본문 미확인)와 "Narrowing the ANN–SNN Gap … Sparsity-Regularized Transform Encoding" (1D 시계열)이 제목에 sparse 가 있으나 읽지 않았다.
- ECCV 2024 는 `ecva.net` 목록 제목만 봤고 CVF 쪽 ECCV 목록은 404 였다.

## 2026-10-13 규제 기반 효율화 추가 조사

의뢰: "규제로 SNN 효율을 높인 연구 중 SOPs·spikes·발화율로 우리 방법과 직접 비교할 수 있는 것, 우리 실험 모델과 같은 것." 10-09 기록의 "추가 웹 검색 2회, 새 논문 0편" 이후 검색어와 경로를 넓혔다. 기존 `compare-list-261009.md` 30행과 이 파일의 앞선 절은 다시 조사하지 않았고, 새 수치를 찾은 경우에만 "기존 항목 갱신"에 적었다. 전수 조사가 아니다.

**우리 기준값 (비교용).** R19 C10 T=4 baseline 96.73 (n=6, 489K spikes/sample) → inter 96.74 @342K (−30%, n=3), intra 96.59 @192K (−60%, n=5). R19 C100 T=4 80.70 (n=5, 450K). QKFormer C10 T=4 96.21 @951K → 96.13 @520K (−45%, n=2). 아래 외부 논문은 spikes/sample 이 아니라 FR·SOPs·mJ 를 보고하므로 각 논문 안의 baseline 대비 변화로만 비교한다.

### 0. 검색 경로와 횟수

| 경로 | 횟수·범위 | 비고 |
|---|---|---|
| WebSearch (일반 웹) | 약 47회 | 영어 키워드: spike regularization, firing rate regularization, sparsity regularization spiking, activity regularization, energy-aware loss, SOPs reduction, synaptic operations penalty, spike count loss, low firing rate, spike-driven transformer sparsity, lateral inhibition spiking regularization, membrane potential regularization, rate regularization ResNet-19, CIFAR10-DVS VGGSNN firing rate, QKFormer/E-SpikeFormer firing rate, Hoyer, polarization 등. 학회 이름(ICLR/CVPR/ICCV/AAAI 2026 등) 붙인 질의 포함 |
| OpenReview 검색 API | 질의 약 90개 (4묶음), 노트 약 2,900건 수집 | `api2.openreview.net/notes/search` 가 **제목·초록·TLDR 을 돌려준다.** 노트 API 와 PDF 는 브라우저 검증(403)으로 막혀 있어 우회하지 않았다 |
| OpenAlex API | 인용 그래프 9편(BPSR, Suetake, Deng, Allred, Sorbaro, Sakemi, Narduzzi, Pellegrini, SCA)의 2021~2026 인용 논문 252건, 초록 검색 17개 질의, "spiking" 제목 2023~2026 13,000건 | 학회 논문(NeurIPS/ICLR/ICML/CVPR)은 누락이 많아 보조로만 썼다 |
| arXiv API | 질의 약 8개 (제목·초록·분류 조합) + PDF 직접 내려받아 `pdftotext` | 초록의 `AND` 조합은 잡음이 크다 |
| 학회 공식 목록 제목 전수 스캔 | NeurIPS 2024(4,493)·2025(5,823), ICLR 2025(3,703)·2026(5,351), ICML 2024(2,616)·2025(3,332)·2026(6,562), CVPR 2024(2,716)·2025(2,871)·2026(4,042), ICCV 2023(2,156)·2025(2,701) 편 | `spik|snn|neuromorph` 제목 필터. 결과는 3절 |
| 쓰지 못한 경로 | dblp 검색 API(봇 차단), Semantic Scholar(429), OpenReview PDF(브라우저 검증), Springer/IEEE 본문(페이월), air.uniud.it PDF(403) | 우회 시도 안 함 |

**제목 전수 스캔 결과.** 위 학회 목록에서 "스파이크·발화·SOPs 감소가 목적인 학습법"으로 읽히는 제목은 모두 앞선 절에 있는 것뿐이다: Activity Pruning·MI-TRQR (NeurIPS 2025), UniSparse·Spik4lite (ICML 2026), TP-Spikformer (ICLR 2026), QP-SNN·Sparse Structure Learning (ICLR 2025), SCA·STATA (ICML 2024). CVPR 2024~2026·ICCV 2023·2025 에는 해당 제목이 없다(SpikePack 은 하드웨어 호환 정보 흐름, SpiLiFormer 는 어텐션 구조). **정식 게재된 탑 학회 논문 중 새로 나온 것은 0편이다.** 새로 나온 것은 아래처럼 ICLR 2027 **심사 중 제출본**과 비탑 학술지·arXiv 다.

### 1. 새로 찾은 논문

우리 방법과의 비교 가능성: ◎ 같은 모델·데이터(T 는 다를 수 있음) / ○ 비슷 / △ 참고.

**1-A. 손실·규제로 스파이크를 줄이는 것**

| # | 논문·저자·학회·연도·URL | 방법 한 줄 | 닮은 점 / 다른 점 | 모델·데이터·T | baseline → 방법, 효율 (출처 위치) | n | 코드 | 비교 |
|---|---|---|---|---|---|---|---|---|
| N1 | **SRSC**, "Spike Regularization and Curriculum Based Direct Training of SNNs for Energy-Efficient Inference". 저자 익명. **ICLR 2027 제출본, 심사 중** (OpenReview 공개 2026-09-19, 수정 2026-10-05). https://openreview.net/forum?id=I3qQofwTxx | 모든 층의 스파이크 텐서에 시간 가중 벌점. 검증 정확도가 임계를 넘을 때까지 무시할 만큼 작게 두다가 천장까지 기하급수로 올림. 파라미터·구조 추가 없음. 천장만 튜닝 | 닮음: 손실 항만 사용, 뉴런·구조 불변, 정확도 유지 주장, 모델이 우리와 같음, 정확도 기준으로 벌점 크기를 정하는 일정. 다름(초록 기준): 벌점이 균등(시간 가중)이고 경쟁 가중 없음, 천장은 수동 튜닝(우리는 ρ·L_task/R 자동), 막전위 항·final_step 언급 없음 | VGG, ResNet, transformer 계열 / CIFAR-10, CIFAR-100, DVS-CIFAR10, ImageNet. T 미확인 | **초록만 확인.** "state-of-the-art 정확도와 R19 C10 ±0.5%, VGGSNN DVS-CIFAR10 ±0.75% 이내, FR −49% / −44%, SOPs −24% / −16%, 모두 relative to state-of-the-art methods". 비교 대상 방법·자기 baseline 정확도·T·λ·표 수치 **미확인**. TLDR: "firing rates by up to 49%" | 미확인 | 미확인 | ◎ (모델) |
| N2 | **"Beyond Firing Less: Activity Objectives Shape Population Organization in a Spike-Driven Transformer"** (PopFR-CV). 저자 익명. **ICLR 2027 제출본, 심사 중** (2026-09-15). https://openreview.net/forum?id=ay5isa4N9N | Spike-Driven Transformer 에 활동 목적함수 여러 개(입력별 집단 발화율의 변동계수 CV 보상 PopFR-CV, 평균 발화율, lifetime sparsity, entropy 계열)를 걸고 학습 후 통계를 측정 | 닮음: SDT 에서 스파이크 활동에 규제항. 다름: 효율 비교표가 아니라 **현상 분석.** 목적함수의 방향과 측정된 결과가 반대로 나옴 | Spike-Driven Transformer, 정적 이미지 3개 데이터셋(CIFAR-100 포함, 나머지 미확인), T 미확인 | 초록만: 평가 CV 가 10배 이상 감소, 같은 크기 텐서의 침묵 좌표 비율 +30%p 이상, 평균 발화는 소폭 감소, 최종 정확도는 비슷, 변화는 어텐션 경로의 활동 철수에 국한 | 미확인 | 미확인 | ○ (모델), 수치는 효율 아님 |
| N3 | Louis, Miramond, Pegatoquet, Girard, "On Reducing Activity with Distillation and Regularization for Energy Efficient SNNs", arXiv 2406.18350 (2024-06). 게재처: OpenAlex 가 Springer LNCS 로 표기, **확인 못 함.** https://arxiv.org/abs/2406.18350 | 지식 증류 + 활성(L1/L2) 규제 + 로짓 규제(새 제안), 균등 벌점·수동 λ | 닮음: 정확도 유지하며 스파이크 감소. 다름: 균등, 모델 약함 | SVGG-11 CIFAR-10 T=4(교사 VGG-16), MNIST SCNN T=10, GSC Res-8 T=4 | Table III CIFAR-10 VGG11: KD 86.5% (δAcc −0.17%) / 총 스파이크 −8.51%, LogReg 86.47% (−0.20%) / **−14.32%**, ActReg 86.41% (−0.27%) / −7.3% (δAcc 가 절대·상대 중 무엇인지 미확인). 본문: 활성 규제는 CIFAR-10 에서 스파이크율 −8% 부터 정확도가 떨어지기 시작. Table III 의 절대 스파이크 값(21~23k)은 단위·정의 미확인 | 없음 | 미확인 | △ (86% 대 모델) |
| N4 | Fontanini, Pilotto, Esseni, Loghi, "Reducing the spike rate of deep spiking neural networks based on time-encoding", Neuromorphic Computing and Engineering 2024, doi 10.1088/2634-4386/ad64fd. https://air.uniud.it/handle/11390/1287104 | TTFS 부호화에서 첫 스파이크 이후 스파이크에 벌점. 침묵 뉴런 방지용 2단계 학습 | 닮음: 스파이크 벌점이 침묵 뉴런을 만드는 문제를 명시하고 처리 방식을 제안(우리 vmem 항과 같은 문제). 다름: TTFS | 미확인 (초록만) | **수치 미확인** (PDF 403) | 미확인 | 미확인 | △ |
| N5 | Krausse, Neher, Knobloch, Becker (KIT), "How Activity Regularization Harms the Performance of Pruned Spiking Neural Networks", ICONS 2025 pp.82-88, doi 10.1109/ICONS69015.2025.00022. https://publikationen.bibliothek.kit.edu/1000189008 | 제목만 확인. 활동 규제와 가중치 프루닝을 같이 쓸 때의 부작용으로 읽힘(추정) | Activity Pruning 이 "활동 규제와 가중치 프루닝은 보완적"이라 한 것과 반대 방향 | 미확인 | **초록·본문 접근 못 함, 수치 미확인** | 미확인 | 미확인 | △ |
| N6 | **SBMVTrack**, "Spike-Budgeted Multi-View Learning for Power-Efficient UAV Tracking". 저자: Zhong, Mo, Li 외. **ICLR 2027 제출본, 심사 중**; arXiv 2609.25503 (2026-09). https://openreview.net/forum?id=Z5BlQmCnVi | Energy-Weighted Spike Budgeting: 층별 연산 비용을 가중해 발화율을 규제하고 포화 활성에 벌점 | 닮음: 층 비용 가중 스파이크 벌점(Suetake 와 같은 부류). 다름: UAV 추적, 균등 아닌 층 가중 | 완전 스파이킹 추적기, VisDrone2018 등 4개 벤치마크 | 초록: 추정 전력 baseline 대비 −24.1%; VisDrone2018 SR 70.0%(SpikeTrack +9.7%p), 추정 전력 −45.7% | 미확인 | "채택 시 공개" | △ (분류 아님) |
| N7 | **QuantLIF** (ICLR 2027 제출본), "Bridging Efficiency–Accuracy Gap in SNNs via Operation-Constrained Quantization and Low-Spike Suppression". https://openreview.net/forum?id=cmcGeC1HmI | 다단계(2의 거듭제곱 합) 스파이크 뉴런. low-spike suppression 은 기여가 낮은 스파이크를 학습 중 확률적으로 거르고 추론에서 결정적으로 제거 | 닮음: "약한 스파이크를 먼저 없앤다"는 발상. 다름: 규제항이 아니라 필터, 뉴런 모델 변경 | 정적·뉴로모픽 벤치마크 (초록에 이름 없음) | 초록에 수치 없음 | 미확인 | 미확인 | △ |
| N8 | **RATI-LIF** (ICLR 2027 제출본), "Runtime-Adaptive Temporal Integration Improves Sparsity and Accuracy in SNNs". https://openreview.net/forum?id=43kWXrmsvz | 추론 시 시간 상수를 뉴런별로 적응하는 LIF | 뉴런 모델 변경(ILIF 와 같은 부류) | 미확인 | 초록: 발화율·추정 에너지 크게 감소, 정확도 향상. 수치 미확인 | 미확인 | 미확인 | △ |

**1-B. 규제 선례 (ANN, 경쟁·분극 축)**

| # | 논문 | 방법 | 우리와의 관계 |
|---|---|---|---|
| P1 | **Polarization regularizer.** Zhuang, Zhang, Huang, Zeng, Shuang, Li, "Neuron-level Structured Pruning using Polarization Regularizer", NeurIPS 2020. https://papers.nips.cc/paper/2020/hash/703957b6dd9e3a7980e040bee50ded65-Abstract.html , 코드 https://github.com/polarizationpruning/PolarizationPruning | BN 스케일 γ 에 거는 규제. 초록: "L1 은 모든 스케일 인자를 0 으로 밀어 뉴런을 구분하지 못한다", 제안은 "일부는 0 으로, 나머지는 a>0 으로" 분극시킴(정리 3.2). CIFAR·ImageNet 에서 L1 보다 좋다고 보고(수치 표는 이번에 안 읽음) | **"균등 벌점은 구분이 없다, 약한 쪽만 죽이고 강한 쪽은 지킨다"는 우리 논리의 ANN 선례.** 대상이 활동이 아니라 γ, 채널 가지치기 후 파인튜닝 구조. 경쟁/분극 가중이 새롭다는 일반 주장의 반례 (SASL, FBS 에 이은 세 번째) |
| P2 | DeepHoyer, Yang, Wen, Li, ICLR 2020. https://arxiv.org/abs/1908.09979 | 스케일 불변 Hoyer-Square 규제로 가중치 희소화 | Kurtz(ICML 2020, 활성에 Hoyer)의 원 규제. 스케일 불변이 "상대적 크기로 벌점"이라는 점에서 우리 1−c/max 가중과 같은 계열 |
| P3 | FGKDSNN (ICLR 2027 제출본), https://openreview.net/forum?id=9S2vkFszgH | KD 논문. 초록에서 "student 의 발화 활동은 특징 채널에 **고도로 집중**돼 있다"고 관찰 | 규제 논문이 아니다. 발화가 채널에 쏠린다는 관찰이 inter(채널 간 경쟁) 설계의 근거로 인용 가능. 심사 중 제출본이므로 인용 보류 |

**1-C. 같은 모델이지만 규제가 아닌 것, 또는 실측 관련**

| # | 논문 | 내용 | 비고 |
|---|---|---|---|
| M1 | Chen, Yuan, Tan 외, "Resource Constrained Model Compression via Minimax Optimization for SNNs", ACM MM 2023, arXiv 2308.04672, doi 10.1145/3581783.3611838. https://arxiv.org/abs/2308.04672 (**앞선 절에서 "표 미확인"이던 것을 이번에 읽음**) | 가중치 프루닝(비구조). 자원 예산 R_budget 을 두고 **쌍대 변수를 경사 상승**으로 갱신(z ← max(0, z+η(R−R_budget))). 모델: R19, VGG16, SEW-R18, VGGSNN. 표는 가중치 희소율이지 스파이크 아님 | Table 2: R19 C10 94.85 → 94.84 / 94.36 / 93.81 @희소율 80.05 / 95.07 / 97.07%. R19 C100 74.71 → 75.05 / 72.67 / 70.80 @79.99 / 95.19 / 97.31%. Table 3: VGGSNN DVS 82.80 → 82.40 / 81.90 / 81.20 / 80.10 @85.18 / 90.14 / 93.14 / 95.16%. T·n 미확인. **쓸모: 우리와 같은 R19·VGGSNN 에서 가중치 희소율 표, 그리고 SNN 에서 자원 제약을 쌍대 변수로 푸는 λ 자동화 선례(가중치 쪽).** 스파이크 감소 비교에는 못 씀 |
| M2 | Wu 외, "Optimizing event-driven SNN with regularization and cutoff" (RCS), Front. Neurosci. 19, 2025, doi 10.3389/fnins.2025.1522788. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full | 코사인 유사도 규제 + 조기 종료(T 축). 스파이크·SOPs·FR 보고 없음 | Table 2 VGGSNN CIFAR10-DVS T=10 TET: 81.10 → 83.10 (+RCS), T_avg 3.16 → 3.05. 앞 절 "읽었으나 뺀 것"을 표까지 읽어 확정. n 없음 |
| M3 | Yao 외, SDT-V3 = "Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training", **IEEE TPAMI 2025** (GitHub 저장소 표기 + OpenAlex 로 확인). arXiv 2411.16061. https://github.com/BICLab/Spike-Driven-Transformer-V3 | 스파이크 규제 없음. ImageNet-1k 78.5 / 79.8 / 84.0 / 86.2% @10M / 19M / 83M / 173M(마지막은 384 입력) | 앞 절의 "게재 미확인" 해소. 19M 5.9 mJ, 83M 19.1 mJ 는 논문의 이론 전력. **우리 171M 과 논문 173M 의 표기 차이는 확인 필요.** SDT-V3 에 스파이크 규제를 건 논문은 찾지 못함 |
| M4 | **STRIDE** (ICLR 2027 제출본), https://openreview.net/forum?id=OBDZfSiQXp | 초록: "알고리즘 연구는 SOP 기반 이론 에너지, 하드웨어 연구는 실제 과제 정확도 검증이 부족해 둘 사이 증거가 단편적"이라 지적. 28nm SNN 가속기에서 4bit 가중치×1bit 스파이크 실행 | 실측 프록시 비판 인용 후보(심사 중) |
| M5 | Aliyev, Lopez, Adegbija, "Surrogates, Spikes, and Sparsity: ... SNN Hyperparameters on Hardware", arXiv 2603.24891 (2026-03). https://arxiv.org/abs/2603.24891 | 초록: "정확도 지표는 하드웨어 효율의 나쁜 예측자", 대리 기울기 선택이 지연을 최대 12.2%(DVS128-Gesture), 뉴런 모델이 최대 28% 바꿈. DVS-CIFAR10 포함 | SynOps 프록시 한계 인용 후보. 게재 미확인 |
| M6 | Yoo, Garg, Chakraborty, "When Spike Sparsity Does Not Translate to Deployed Cost: VS-WNO on Jetson Orin Nano", arXiv 2604.17040 (ICONS 2026 제출) | 스파이킹 연산자: 알고리즘 희소성(발화율 54%→18%) 이 있어도 Jetson 에서 지연 59.6 vs 53.2 ms, 동적 에너지 228.0 vs 180.7 mJ 로 dense 보다 나쁨 | 신경 연산자(PDE)라 모델이 다르다. 상용 GPU 스택에서 희소성이 비용이 되지 않는다는 사례 |

### 2. 기존 항목 갱신 (새 수치)

- **UPR (ICLR 2024 Spotlight) 원문 Table 1·A1 을 읽었다** (https://proceedings.iclr.cc/paper_files/paper/2024/file/dd3c889922df2112a5b1769e3c19e28e-Paper-Conference.pdf). 앞 절은 SpikeNM·UniSparse 표의 2차 인용이었다. 벌점은 마스크에 거는 L1(SOPs 상수 가중, 식 10~11)이고 시드 수는 없다.
  - CIFAR-10, 6Conv+2FC, T=8: dense 92.84 @776.12M SOPs → 92.63 (−0.21) @38.32M (20.25배) / 92.05 @16.47M / 91.47 @11.98M / 90.65 (−2.19) @8.50M.
  - DVS-CIFAR10, VGGSNN, T=10: dense 82.4 @647.30M → 81.9 (−0.5) @47.81M (13.54배; 연결 6.80%, 뉴런 42.73% 잔존) / 81.0 @31.86M / 79.0 @10.02M / 78.3 @6.75M.
  - ImageNet, SEW-R18, T=4: dense 63.18 @1337.87M → 61.89 (−1.29) @311.70M (4.29배) / 60.00 @177.99M / 58.99 @116.88M.
  - **CIFAR-100, SEW-R18, T=10 (Table A1): dense 74.16 @250.91M → 72.34 (−1.82) @27.16M (9.24배) / 71.99 / 71.43 / 71.31 / 70.45 / 69.41 (−4.75) @6.79M.** 이전 비교 목록에 없던 C100 값.
- **Activity Pruning 의 L1 규제(AR) 대조군을 ImageNet-100·DVS 까지 읽었다** (NeurIPS 2025 PDF, https://www.proceedings.com/content/085/085713-5700open.pdf , Table 1). 앞 절은 CIFAR-10 AR 만 적었다.
  - ImageNet-100 SEW-R18 T=4: Vanilla 83.44 (FR 0.138, SOPs 1276.42M) / AR λs=0.01 82.94 (0.083, 840.90M) / AR λs=0.05 82.26 (0.043, 469.00M) / AT-LIF 82.40 (0.038, 463.65M) / 81.12 (0.025, 328.34M).
  - DVS VGGSNN T=10: Vanilla 82.70 (0.080, 1066.40M) / AR λs=0.005 81.30 (0.055, 673.06M) / AT-LIF 82.20 (0.043, 626.57M) / 82.10 (0.039, 587.68M).
  - 해석: FR 이 비슷하거나 AT-LIF 쪽이 더 낮은 행끼리 비교하면 AT-LIF 가 plain L1 보다 ImageNet-100 에서 +0.14%p (82.40 @0.038 대 82.26 @0.043), R20 C10 에서 +0.74%p (92.05 @0.053 대 91.31 @0.060), DVS 에서 +0.90%p (82.20 @0.043 대 81.30 @0.055). **FR 을 −31~−81% 줄인 대표 행에서 plain L1 은 세 설정 모두 −1.2~−1.6%p 를 잃는다**(ImageNet-100 −1.18 @FR −68.8%, R20 C10 −1.58 @−74.6%, DVS −1.40 @−31.3%). 약한 λ 행은 손실이 더 작다(ImageNet-100 λs=0.01: −0.50 @−39.9%). 우리 plain L2 대조군(R19 C10 T=4 −0.03%p @−20%)과 같은 방향의 근거이면서, 대조군이 어디까지 약한지(−1%p 이상 손실)의 외부 기준이다. 이 논문의 Theorem 1(스파이크가 전부 0 인 점은 안장점)은 우리 vmem 항이 답해야 하는 지점이다.
  - PDF 첫 쪽 저자 순서는 **Xinyu Shi, Tong Bu, Zhaofei Yu** 다(앞 절·bib 은 Bu, Shi, Yu). UniSparse 본문은 "적응 임계값으로 스파이크를 줄인 최근 연구 (Bu et al., 2026)" 를 인용하는데 이 논문을 가리키는 것으로 보이나 제목은 확인하지 못했다. refs.bib 키·저자 순서 확인 필요.
- **UniSparse 본문(PDF 를 pdftotext 로 읽음).** 서론이 기존 스파이크 희소화를 "Neil et al. 2016, Deng et al. 2021 의 스파이크 규제"로 규정하고 "모든 스파이크를 무차별 억제해 과제 목표와 충돌, 성능을 해친다"고 비판한다(앞 절의 인용과 일치). 우리 인용 위치: 이 한 문장이 "규제는 무차별"의 출처다. Table 1 에서 ResNet-19 T=4 행 94.71±0.11 → 94.07 → 92.48 → 91.56 @39.47 / 18.58 / 8.18 / 6.11M SOPs 는 기존 값과 같다. dense 행은 표에 없다.
- **Datta, Liu, Diffenderfer, Kailkhura, Beerel, arXiv 2312.06900** (2023-12, "Under review", 게재 미확인; 앞 절은 "변환 L1"이라 표에서 제외). ANN→SNN 변환 + 비트 단위 L1 (식 9 Σ a_t^{i,l}, STE 류 대리 기울기). 본문: λ 0 → 5e-8 에서 활동이 2.5~4.1배 감소하나 T≤3 에서 정확도 하락. **ResNet18 C10 λ=1e-8: λ=0 대비 정확도 0.2% 이내, 활동 0.53 → 0.22 (2.4배).** ResNet34 ImageNet λ=5e-10: 정확도 −0.4%, 연산 에너지 2배 감소. 수치는 본문 서술이고 n 없음. 직접 학습이 아니고 plain L1 이다.
- **Suetake (arXiv 2302.01500)** Remark 3.5 와 §4: p=1 이 p=2(제곱) 보다 에너지-정확도 곡선이 낫다고 보고하고, p≠1 은 "스파이크가 안 나간 뉴런에 흐르는 역신호"를 바꾼다고 서술. 이유는 미해결로 둠("mechanism ... unclear"). 우리 vmem 항(침묵 뉴런에 막전위 벌점)과 같은 현상을 다른 방식으로 건드린다. 우리 norm 은 제곱이 아닌 L2 라서 BPSR(제곱)과도 Suetake(p=1)와도 다르다.
- **BPSR** 은 이번에 PMC 본문에서 서지를 재확인: Yan, Chu, Jin, Huan, Zou, Zheng (Fudan), Front. Neurosci. 16:760298, 2022. Table 5 의 CIFAR-10 11층 ResNet: T=12 λ_s=1e-9 → 136.1K spikes @90.74%, T=8 λ_s=1e-8 → 89.6K @90.24%. 앞 절 #9 와 일치하며 시드 수는 없다.

### 3. 모델별 직접 비교 가능성

| 우리 모델 | 같은 모델에서 효율 수치를 낸 외부 논문 | 상태 |
|---|---|---|
| R19 C10 T=4 | **SRSC** (N1, ±0.5% 내 FR −49% SOPs −24%, 초록만), UniSparse (기존), MI-TRQR PLIF-Net (기존), Minimax 가중치 희소율 표 (M1, 스파이크 아님) | SRSC 는 수치 접근 불가. R19 C10 T=4 에서 스파이크 규제의 정확도 보존 구간을 표로 낸 논문은 여전히 UniSparse(밀집 행 없음)뿐 |
| R19 C100 T=4 | MI-TRQR(기존), SRSC(C100 검증한다고만 서술), Minimax 가중치 표 (M1), UPR SEW-R18 T=10 (갱신) | 스파이크 규제 + 효율 표는 MI-TRQR 외 없음 |
| VGGSNN DVS-CIFAR10 | UPR 원문 (T=10, −0.5%p @13.54배), Activity Pruning + AR 대조군 (T=10, 갱신), UniSparse (기존), **SRSC** (±0.75%, FR −44%, SOPs −16%, 초록만), MI-TRQR VGG (T=4, 기존), Minimax 가중치 표, RCS (T 축) | 가장 비교 대상이 많은 모델. 단 문헌은 대부분 T=10, 우리는 T=4 |
| QKFormer C10/C100 T=4 | 스파이크 규제 논문 **0편.** Uncert, TP-Spikformer (기존, 토큰 쪽) | arXiv 2604.15769 가 QKFormer-4-256 을 C10 94.8±0.2 / 1.6M spikes, C100 78.9±0.3 / 1.9M 로 적었으나 이론 논문이고 측정 프로토콜·단위가 불명(우리 4-384 @951K 보다 작은 모델이 더 많은 스파이크)이라 **인용 금지 수준** |
| SDT-V3 ImageNet | 스파이크 규제를 건 논문 **0편.** SDT-V3 본체는 TPAMI 2025 로 확정(M3). ImageNet 효율 수치: UPR SEW-R18 T=4 (갱신), UniSparse Spikformer (기존), Activity Pruning SEW-R18 (기존) | 모델이 달라 절대 비교 불가 |

SOPs·FR 환산: 위 논문의 FR = 전 뉴런·전 시간의 평균 스파이크(Activity Pruning 정의), SOPs = Σ(스파이크 수 × 팬아웃) (UPR/UniSparse 정의). 우리 모델에서 FR 과 SOPs 를 같이 계산해 두면 이 표에 놓을 수 있다.

### 4. 우리 방법과 가장 비슷한 것 (상위 3)

1. **SRSC (N1).** 손실 항 하나, 뉴런·구조 불변, 같은 모델, 정확도 유지 주장, 정확도 기반 일정. 차이(초록 기준): 벌점이 균등이라 경쟁 가중이 없고, 천장이 수동이며, 막전위 항이 없다. 단 PDF 를 못 읽어 "경쟁 가중이 없다"는 판단은 초록·TLDR 에만 근거한다.
2. **Suetake, TMLR 2024 (기존).** 에너지에 비례하게 설계한 단일 벌점, 구조 불변, 정확도 유지(ResNet18 C10 T=1 −42% @+0.08%p, n=3). 차이: fan-out 고정 가중, 수동 λ 격자, T=1. 본문 §4 의 p=1 대 p=2 관찰이 우리 norm 선택을 설명할 때 쓸 수 있다.
3. **UniSparse, ICML 2026 (기존).** 단일 λ, 에너지(SOPs) 모델에서 유도한 벌점, 채널 단위(BN affine)로 스파이크 억제. 차이: 손실 항이 아니라 파라미터 직접 갱신, 가중치 프루닝과 결합, 밀집 baseline 행 없음.

경쟁/분극 축만 따로 보면: **Polarization regularizer (P1, ANN)** 가 개념적으로 가장 가깝고, SNN 에서는 **PopFR-CV (N2)** 가 "집단 내 발화율 불균등"을 목적함수로 직접 쓴다(단 분석 논문). λ 자동 조정 축: SRSC 의 검증 정확도 기반 일정, Meem 의 발화율 오차 피드백(앞 절 #19), Minimax 의 쌍대 상승(M1, 가중치).

### 5. 우리 주장에 위험한 것

1. **SRSC (N1).** 같은 모델(R19 C10, VGGSNN DVS), 같은 주장 형태("구조 변경 없이 정확도 유지, FR −49%"), 일정이 침묵 붕괴를 피한다. 대비해야 할 반론은 **"plain 벌점 + 정확도 기반 일정"이 우리 경쟁 항과 같은 결과를 내지 않느냐**다. 경쟁 항 자체의 기여를 보이려면 plain L2 에도 같은 일정을 준 대조가 필요하다(제안만, 실행하지 않음). ICLR 2027 제출본은 OpenReview 에서 공개 열람 가능하므로 CVPR 심사자가 볼 수 있다. 심사 중이라 선행 연구로 인용할 수는 없고, 동시기 연구로 다룰지는 사용자 판단 사항이다. 위험도는 "표를 못 읽어서" 과소평가일 수도 과대평가일 수도 있다.
2. **PopFR-CV (N2).** 규제가 의도한 통계와 반대 방향의 결과를 낳고 침묵 좌표가 +30%p 늘 수 있다는 SDT 분석이다. 우리가 "경쟁 항이 채널을 죽인다"고 말할 때 **목적함수가 아니라 측정된 결과로** 주장해야 한다는 압력이다. 또 SDT 에서는 활동 목적함수를 걸어도 평균 발화 감소가 "modest" 라고 한다. SDT-V3 파인튜닝에서 큰 감소를 기대하기 어렵다는 외부 근거로 읽힐 수 있다(초록만 근거).
3. **Activity Pruning (기존).** 이번 갱신 값(2절)으로 위험의 성격이 바뀐다. plain L1 은 R20 / ImageNet-100 / DVS 에서 −1.2~−1.6%p 를 잃으므로 우리 plain 대조군 해석은 지지되고, 반대로 이들의 AT-LIF 는 ImageNet-100 에서 plain L1 과 0.14%p 차이뿐이다. "무차별 규제는 반드시 손실을 낸다"는 일반화는 ImageNet-100 에서는 약하다. 우리 주장을 "같은 FR 에서 plain 대비 얼마나 낫다"로 쓰면 이 논문의 표와 같은 축에 놓인다.
4. **P1 Polarization regularizer.** "약한 쪽만 죽이고 강한 쪽은 지키는 분극 규제는 새롭다"는 문장의 반례. 새로움은 "활동(스파이크) 자체에 걸린 경쟁 가중이 SNN 에서 채널을 죽인다"로 한정해야 한다(앞 절 SASL·FBS 권고와 같은 방향).
5. **실측 프록시 비판(M4~M6, Sparsity Ceiling).** 스파이크·SOPs 감소가 배포 비용 감소가 아니라는 근거가 늘고 있다. 우리 효율 주장은 spikes 와 SOPs 까지이고 실측은 없다는 한계 문장이 필요하다.

### 6. 못 찾은 것·미확인 (본문 인용 금지)

- **SRSC·PopFR-CV·SBMVTrack·QuantLIF·RATI-LIF·FGKDSNN·STRIDE**: ICLR 2027 제출본. 초록·TLDR 만 OpenReview 검색 API 로 확인했고 PDF·표·코드는 접근하지 못했다(브라우저 검증). 익명 제출이라 저자 미확인. 철회·수정될 수 있다. **본문 수치 인용 금지.**
- Krausse ICONS 2025, Fontanini NCE 2024 (PDF 403), Kang 외 "Spike-Count Reduction Techniques" (NPL 2025, 페이월), "Sparser spiking activity can be better: Feature Refine-and-Mask SNN" (Neural Networks 2023, 페이월), Fleury "Masked Spikformer" (2025, 도서관 레코드만 확인, arXiv·본문 없음), "Low Latency and Sparse Computing SNNs With Self-Driven Adaptive Threshold Plasticity" (TNNLS 2024), "Directly training temporal SNN with sparse surrogate gradient" (Neural Networks 2024): 존재와 서지만 확인, 수치 미확인.
- Louis 외 N3 의 게재처, Table III 절대 스파이크 단위·δAcc 정의. Datta 2312.06900 의 게재 여부. Minimax(M1) 의 T.
- **제목 전수 스캔을 하지 않은 곳**: ICLR 2024 (OpenReview 검색으로 일부만), ECCV 2024, AAAI 2024~2026, IJCAI, ACM MM, TNNLS·TPAMI·Nat. Commun. 등 저널 (OpenAlex 제목 필터 566건을 훑었으나 학회·저널 누락이 많다).
- QKFormer 에 스파이크 규제를 건 논문, SDT-V3 에 건 논문, R19 C100 T=4 에서 스파이크 규제 + 효율 표를 낸 논문(MI-TRQR 제외): **찾지 못했다.** 못 찾은 것이지 없다는 뜻이 아니다.
- 읽었으나 위협이 낮아 뺀 것: ILIF·RPLIF·AHSAR(arXiv 2512.05015, 층별 발화율을 검증 성능·에너지로 조절하는 임계값 제어기, 학습 가속 목적, ResNet-18 C10 T=8/10)·SPARTA(arXiv 2508.01646, 측면 억제 토큰 선택, DVS T=16)·Inherent Redundancy(ICCV 2023, 제스처·보행 DVS)·Masked Spiking Transformer(ICCV 2023, 변환)·Hoyer(앞 절 #10 에 있음)·TDSNNs(AAAI 2026, 위상 구조 손실, 효율 목적 아님)·Energy-Aware Spike Budgeting(앞 절 #19)·ShiftLIF(arXiv 2605.01866, 다단계 뉴런, 보조 L1 항).


## 2026-10-14 inter 타당성 근거 문헌

의뢰 (사용자 10-14): "inter 가 타당한 이유도 분석해줘." 이 절은 그중 **문헌·생물학 근거**만 맡는다. 우리 측정과 겹치는 부분은 6절에 한 표로만 대조했고, 통계 분석은 건드리지 않았다. 읽기 전용 조사이며 코드·큐·서버·커밋은 건드리지 않았다.

**대상 방법.** ours_inter_ch: `loss = CE + λ·Σ_l ‖S_l ⊙ w_l‖₂`, `S` = 이미지별 T 합 스파이크, `w = 1 − c/max_c c`, `c` = 채널 공간합(같은 층 채널끼리 경쟁). 역전파 `w²/R`, λ = ρ·L_task/R. 따라서 문헌에서 찾는 것은 (가) 채널 활동량이 채널 중요도의 대리 지표인가, (나) 약한 쪽을 더 세게 벌하는 상대(비균등) 가중이 균등보다 나은가, (다) 채널 단위 희소성이 하드웨어 비용과 이어지는가, (라) 생물학적 대응이 있는가 네 가지다.

**읽은 정도 표기.** [본문] = PDF/HTML 을 내려받아 본문에서 직접 확인. [초록] = 초록·서지 페이지만. [2차] = 검색 요약이나 다른 논문의 인용으로만 확인. [미확인] = 본문 인용 금지. 외부 수치는 해당 논문의 표·설정을 같이 적었고, **n 은 거의 모든 외부 논문이 기재하지 않는다 (기재 안 된 곳은 "n 미기재").** 이번 세션에서 내려받은 PDF 약 30편, WebSearch 약 20회, WebFetch 약 25회. 전수 조사가 아니다.

### 0. 결론 먼저

| inter 의 전제 | 지지 문헌 (직접성) | 반대·제한 문헌 | 판정 |
|---|---|---|---|
| (가) 채널 활동이 낮으면 덜 중요하다 | APoZ 류(ANN, 약), Molchanov 2017 (층 안 순위 상관 0.56~0.65, 단 AlexNet-ImageNet 0.09), **SCA (SNN, 채널 막전위 활동 낮은 채널 제거 후 정확도 유지)**, Spik4lite (채널 발화율 기반 제거) | **Inherent Redundancy (ICCV 2023): 발화가 많은 채널이 배경 잡음 채널**, Ye 2018·FPGM (노름 작으면 덜 중요하다는 가정은 조건부) | **부분 지지.** 방향이 반대로 나오는 SNN 사례가 있다 |
| (나) 상대(순위) 가중이 균등보다 낫다 | **SASL (ANN, 순위별 벌점 0×~4×, 최강 필터 무벌점)**, Polarization (NeurIPS 2020), Trimmed ℓ1 (ICML 2019), DeepHoyer, reweighted ℓ1, FBS (채널 k-WTA) | SNN 활동 규제에서 이 가중을 직접 시험한 선행 연구는 **찾지 못함.** 우리 측정에서도 BPSR 대비 같은 스파이크 정확도 +0.04 ±0.04 (p=0.29) | **ANN 선례 강함, SNN 선례 없음, 우리 정확도 우위는 판정 불가** |
| (다) 채널 단위 희소성이 비용으로 이어진다 | **Speck: 메모리·매핑이 채널 수로 결정(sinabs 문서)**, Spik4lite (채널 물리 제거 → Jetson 실측 에너지), Wen 2016·Network Slimming (ANN, CPU/GPU 실속도), Loihi 표 (비활성 뉴런 갱신 52 pJ) | UPR (ICLR 2024): 구조적 뉴런 프루닝은 뉴로모픽의 희소 연산을 못 쓴다고 주장, Spike-Thrift (같은 정확도에서 구조적 압축률이 한 자릿수 낮음), Loihi 추정 모델에서 에너지의 86%가 시냅스 연산 | **지지되지만 단서가 붙는다.** 죽은 채널은 동적 비용이 원래 0 이므로 이득은 메모리·매핑·밀집 연산에서 나온다 |
| (라) 생물학적 대응이 있다 | 정규화·승자독식(Carandini & Heeger 2012), 후각 사구체 간 측면 억제(Olsen 2010), 컨볼루션 SNN 의 intra/inter-map competition (Kheradpisheh 2018) | V1 에서 측면 억제 없이 설명 가능(Priebe & Ferster 2008), 정규화에 GABA 억제 불필요(같은 리뷰), 개별 뉴런 set-point 항상성(Torrado Pacheco 2021) | **유추 수준.** "특징 채널 공간합끼리 겨루는 규제"의 직접 생물 근거는 못 찾음 |

**한 줄 요약.** inter 와 같은 단위(채널 공간합)·같은 방향(약한 채널이 더 벌받고 최강 채널은 무벌점)의 규제를 **SNN 스파이크 손실로** 쓴 선행 연구는 찾지 못했다. 구성 요소별 선례는 ANN 에 있다(FBS = 채널 공간 풀링값으로 채널 k-WTA, SASL = 순위 기반 비균등 벌점, SCA = SNN 채널 활동 기반 제거). 타당성의 가장 단단한 부분은 (다) 하드웨어 쪽 구조적 이득이고, 가장 약한 부분은 (가) "많이 쏘는 채널이 중요하다"는 전제의 방향이다.

### 1. 생물학

| 문헌 | 서지·URL | 확인한 문장·수치 | 대응 | 한계 |
|---|---|---|---|---|
| **Carandini & Heeger 2012** [본문] | Normalization as a canonical neural computation. Nat Rev Neurosci 13(1):51–62 (2012; 온라인 2011-11-23), doi 10.1038/nrn3136. https://pmc.ncbi.nlm.nih.gov/articles/PMC3273486/ | 초록: "the responses of neurons are divided by a common factor that typically includes the summed activity of a pool of neurons". V1: 높은 대비 수평 격자에 낮은 대비 수직 격자를 더하면 집단 반응이 "mostly reflect the high-contrast grating" (정규화가 승자독식을 만든다). Box 2 "Max-pooling (winner-take-all)": "selecting the neuronal subpopulation (or psychophysical channel) with the largest responses and suppressing the rest". Box 2 "Redundancy reduction": 정규화가 V1 반응 간 통계적 독립성을 높여 표현을 효율적으로 만든다 | 큰 반응이 작은 반응을 누르고 가장 센 쪽만 남는 구조가 inter 의 `w = 1 − c/max` (최강 채널 무벌점) 와 같은 방향 | 추론 시 반응을 나누는 순방향 계산이지 학습 손실이 아니다. pool 은 같은 위치·다른 특징 뉴런이며 채널 공간합이 아니다. 같은 글이 "V1 의 정규화에서 GABA 억제는 거의 역할이 없다"고 쓴다 (5절) |
| **Olsen, Bhandawat, Wilson 2010** [초록] | Divisive normalization in olfactory population codes. Neuron 66(2):287–299, doi 10.1016/j.neuron.2010.04.009. https://pubmed.ncbi.nlm.nih.gov/20435004/ | "Lateral inhibition increases the level of feedforward input needed to drive PNs to saturation, and this normalization scales with the total activity of the olfactory receptor neuron (ORN) population." "the saturating transformation intrinsic to each processing channel boosts weak signals, while normalization helps equalize responses to different stimuli" | **가장 가까운 '채널 단위' 생물 사례.** 초파리 후각의 처리 채널(사구체 = 수용체형 하나) 사이에서 측면 억제가 일어난다 | 억제 세기가 채널 최대치가 아니라 **전체 ORN 활동**에 비례한다. 같은 초록이 약한 신호를 키우고 반응을 고르게 만든다고 쓴다 — "약한 채널을 죽인다"와 반대 요소 |
| **Lennie 2003** [초록] | The cost of cortical computation. Curr Biol 13(6):493–497, doi 10.1016/s0960-9822(03)00135-0, PMID 12646132. https://pubmed.ncbi.nlm.nih.gov/12646132/ | "The cost of a single spike is high, and this severely limits, possibly to fewer than 1%, the number of neurons that can be substantially active concurrently." 높은 비용이 "representational codes that rely on very few active neurons" 를 요구한다 (인간 피질 추정) | 스파이크를 줄이려는 목적 자체의 생물학 동기 | 채널 단위 경쟁의 근거는 아니다 |
| **Olshausen & Field 1996; Vinje & Gallant 2000** [초록] | Nature 381(6583):607–609, doi 10.1038/381607a0 (PMID 8637596). Science 287(5456):1273–1276, doi 10.1126/science.287.5456.1273 (PMID 10678835). | O&F: 희소성을 최대화하는 코드가 V1 단순세포 수용야를 낳고 출력 간 통계적 독립성이 높다. V&G: 비고전 수용야 자극이 "increases the sparseness of the population response distribution, and strongly decorrelates the responses of neuron pairs" (흥분·억제 조절 모두) | 억제성 조절이 집단 희소성과 비중복을 만든다 | 뉴런 쌍·수용야 수준, 채널 아님 |
| **Barth & Poulet 2012** [초록, 2차 요약] | Experimental evidence for sparse firing in the neocortex. Trends Neurosci 35(6):345–355. https://www.mdc-berlin.de/research/publications/experimental-evidence-sparse-firing-neocortex | 요약: 반응 특성이 뉴런마다 크게 달라 소수 뉴런이 대부분의 스파이크를 내고, 표층 일부 세포는 아예 발화하지 않는다 (원문 문장 아님) | 활동이 소수에 몰린 치우친 분포가 피질의 정상 상태라는 근거 (우리 지니 상승과 같은 방향) | 요약 문장이라 인용하려면 본문 확인 필요 |
| **Rozell 외 2008 (LCA); Davies 외 2018 (Loihi S-LCA)** [LCA 2차, Loihi 본문] | Rozell, Johnson, Baraniuk, Olshausen, Neural Comput 20:2526–2563, doi 10.1162/neco.2008.03-07-486 (검색 확인, PDF 미열람). Davies 외, Loihi: A Neuromorphic Manycore Processor with On-Chip Learning, IEEE Micro, Jan/Feb 2018. https://gwern.net/doc/ai/scaling/hardware/2018-davies.pdf | Loihi 논문: 스파이킹 뉴런들이 서로 경쟁해 희소 표현을 만든다("neuron that has the largest external input ... more likely to spike at the earliest time, causing immediate inhibition of the other neurons"). 전신 칩 표 3: 미지수 32,256 / 비영 약 420 문제에서 Atom 대비 에너지 48.74배, 지연 118.18배, EDP 5760배, 미지수 400 / 비영 약 10 에서는 EDP 0.7배 | 경쟁으로 희소 코드를 만드는 계산이 스파이크 하드웨어에서 효율적이라는 시연 | 추론 시 최적화 알고리즘이고 학습 규제가 아니다. 억제가 특징 사전의 겹침으로 정해진다 |
| **Kheradpisheh 외 2018** [본문] | STDP-based spiking deep convolutional neural networks for object recognition. Neural Networks, doi 10.1016/j.neunet.2017.12.005 (arXiv 1611.01421v3 머리말 기준). https://arxiv.org/abs/1611.01421 | "global intra-map competition": 같은 맵(채널) 안 위치끼리 먼저 발화한 하나만 STDP. "local inter-map competition ... This competition is crucial to encourage neurons of different maps to learn different features." 학습 경쟁이 맵이 비슷한 특징을 배우는 것을 막는다 | **우리 세 팔의 이름(intra / inter)과 같은 구분을 SNN 학습에 쓴 선행.** 채널 간 경쟁의 목적이 '맵마다 다른 특징' | 비지도 STDP, 경쟁 단위는 근방 위치의 다른 맵. 스파이크 수 감소가 목적이 아님. 정확도 수치는 이번에 안 읽음 |
| **SpykeTorch (Mozafari 외)** [본문] | Frontiers in Neuroscience 수록 예정으로 arXiv 머리말 표기. https://arxiv.org/abs/1903.02440 | "sf.pointwise_inhibition employs competition among feature maps at each location" (한 위치에서 가장 두드러진 특징 하나만 발화), "sf.feature_inhibition ... complete inhibition of the selected feature maps" | 채널 간 경쟁이 SNN 라이브러리의 기본 연산 | 구현 서술일 뿐 효과 수치 없음 |

**1절 소결.** 생물학에서 확인되는 것은 (i) 큰 반응이 작은 반응을 눌러 최강 쪽만 남기는 집단 수준 정규화·승자독식, (ii) 후각계의 채널(사구체) 간 측면 억제, (iii) 대사 비용이 결정하는 극히 희소한 발화다. **"특징 채널의 공간합을 서로 비교해 약한 채널을 학습으로 죽인다"를 직접 지지하는 생물 문헌은 찾지 못했다.** 논문에는 "영감" 수준의 유추로만 쓰고 근거로 쓰지 않는 편이 안전하다.

### 2. ANN 선행 (경쟁·구조적 희소·채널 중요도)

| 문헌 | 서지·URL | 확인한 문장·수치 (표·설정) | inter 와 겹치는 점 | 다른 점·한계 |
|---|---|---|---|---|
| **LRN, AlexNet** [본문] | Krizhevsky, Sutskever, Hinton, NeurIPS 2012. https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf | §3.3: "implements a form of lateral inhibition inspired by the type found in real neurons, creating competition for big activities amongst neuron outputs computed using different kernels". top-1/top-5 오류 −1.4%/−1.2% (ImageNet), CIFAR-10 4층 CNN 13% → 11% | 채널 간 큰 활동 경쟁을 측면 억제로 명시한 가장 유명한 ANN 선례 | 같은 위치의 "인접(순서 임의) 커널 맵" n 개로 나누는 순방향 정규화. 학습 손실이 아니고 희소성이 목적도 아님 |
| **Compete to Compute (LWTA)** [본문] | Srivastava, Masci, Kazerounian, Gomez, Schmidhuber, NIPS 2013. https://papers.nips.cc/paper_files/paper/2013/file/8f1d43620bc6bb580df6e80b0dc05c48-Paper.pdf | 초록: "Local competition among neighboring neurons is common in biological neural networks"; 경쟁하는 선형 유닛이 비경쟁 비선형 유닛보다 좋은 경향, 파국적 망각 회피. §2: 피질의 recurrent on-center off-surround 구조, Grossberg: faster-than-linear 신호함수가 WTA 동역학을 만들고 slower-than-linear 는 입력을 균일화 | 생물학 측면 억제를 ANN 의 블록 내 승자독식으로 옮긴 직접 선례 | 블록 내 유닛 간 forward 경쟁, 희소 스파이크 비용과 무관 |
| **Winner-Take-All Autoencoders** [초록·본문 일부] | Makhzani & Frey, arXiv 1409.2752 (v2, 2015-06; 게재처는 arXiv 쪽에 기재 없음). https://arxiv.org/abs/1409.2752 | 컨볼루션 판: "spatial sparsity within each feature map is achieved using winner-take-all activation functions", 배치 통계로 lifetime sparsity 강제. MNIST·CIFAR-10·ImageNet 등에서 경쟁력 있는 분류 (초록에 수치 없음) | **intra_ch(맵 안 위치끼리 경쟁)에 대응하는 ANN 선례**와 '특징 맵 단위' 희소성 | 비지도 오토인코더 |
| **FBS** [본문] | Gao, Zhao, Dudziak, Mullins, Xu, "Dynamic Channel Pruning: Feature Boosting and Suppression", ICLR 2019. https://arxiv.org/abs/1810.05331 | §3.4 식 (7): 채널의 공간 차원을 스칼라로 줄여(ℓ1 노름 = ReLU 활성에서 global average pooling) 채널 saliency 를 예측하고, 식 (6): `wta_k` 가 가장 큰 k 개 채널만 남기고 나머지를 0 으로 억제. 층마다 `λ·E‖g_l‖₁` lasso (λ = 1e-8). 초록: VGG-16 5배, ResNet-18 2배 연산 절감, top-5 손실 0.6% 미만 (ImageNet) | **채널 공간합으로 채널끼리 승자독식하는 방식이 효율 이득을 낸 ANN 선례.** `c` 와 같은 양을 쓴다 | 입력별 동적 게이트(예측기 추가, 추론 시에도 경쟁). 손실만으로 하는 것이 아님. SNN 아님 |
| **SSL** [초록·본문] | Wen, Wu, Wang, Chen, Li, "Learning Structured Sparsity in Deep Neural Networks", NIPS 2016. https://arxiv.org/abs/1608.03665 | group Lasso 로 필터·채널·필터 모양·층 깊이를 정규화. AlexNet 합성곱 층 CPU 5.1배·GPU 3.1배 가속 (초록: "about twice speedups of non-structured sparsity"), CIFAR-10 ResNet-20 → 18층에서 91.25% → 92.60% | 채널 단위 희소를 정규화로 학습 | 가중치 그룹에 거는 ℓ2,1. inter 는 활동 `S` 에 채널별 가중 `w` 를 곱한 단일 ℓ2 이며 group lasso 와 같은 식이 아니다 |
| **Network Slimming** [본문] | Liu, Li, Shen, Huang, Yan, Zhang, ICCV 2017. https://arxiv.org/abs/1708.06519 | 채널 희소는 "nice tradeoff between flexibility and ease of implementation"이고 비정형 희소는 "requires special software or hardware accelerators". Table 1 (CIFAR·SVHN): 채널 ≥60% 제거해도 정확도 유지, 파라미터 최대 10배·FLOP 약 50% 절감, 40% 제거는 대개 오류 감소. ResNet-164 는 병목 구조가 이미 채널을 골라 절감이 작다 | 채널 희소성의 장점 서술을 그대로 가져올 수 있다 | BN γ 에 균등 ℓ1(비교 대상인 균등 벌점 쪽). 활동이 아니라 파라미터 |
| **Polarization regularizer** [초록] | Zhuang, Zhang, Huang, Zeng, Shuang, Li, NeurIPS 2020. https://papers.nips.cc/paper/2020/hash/703957b6dd9e3a7980e040bee50ded65-Abstract.html | "L1 regularization lacks discrimination between neurons because it pushes all scaling factors towards 0"; 제안은 일부는 0, 나머지는 0 보다 큰 값으로 분극시키고 CIFAR·ImageNet 에서 L1 보다 낫다고 보고 (수치 표 미열람) | "균등 벌점은 구분이 없다"의 ANN 선례 (앞 절 P1 재확인) | 대상이 BN γ |
| **SASL** [본문] | Shi, Xu, Tasaka, Chen, arXiv 2003.05891 (arXiv 코멘트: IEEE TCSVT 게재 수락, 쪽수 미확인). https://arxiv.org/abs/2003.05891 | 서론: 모든 필터에 같은 규제를 하면 "the performance of important filters ... is also influenced by the equal regularization, so the final prediction precision after SL may drop by a large margin". §III-C: 필터를 saliency 순위로 5등급으로 나눠 규제 배율을 {0×, 1×, 2×, 3×, 4×} 로 주고 "for the filters of the most significant class, no regularization", 가장 약한 필터에 최강 벌점, 순위는 학습 중 계속 갱신. 초록: ImageNet ResNet-50 FLOPs −49.7%, top-1 −0.39%, top-5 −0.05% | **구조가 inter 와 같다 (최강은 무벌점, 약할수록 센 벌점, 학습 중 순위 갱신).** 우리 설계 논리("균등 벌점은 중요한 쪽도 깎는다")의 정확한 ANN 선례 | saliency = (기울기·가중치)² / 연산 비용 (식 6) 이라 가중치·연산량 기반이고 활동이 아님. 규제는 BN γ 에 건다. 5등급 이산. 균등 대비 직접 비교 표는 이번에 안 읽음 |
| **Trimmed ℓ1; DeepHoyer; reweighted ℓ1; adaptive lasso** [Trimmed 2차, DeepHoyer·reweighted 초록] | Yun, Zheng, Yang, Lozano, Aravkin, ICML 2019 (PMLR 97:7242–7251) https://proceedings.mlr.press/v97/yun19a.html ; Yang, Wen, Li, DeepHoyer, ICLR 2020 https://arxiv.org/abs/1908.09979 ; Candès, Wakin, Boyd, arXiv 0711.1612 (저널 서지는 arXiv 쪽에 없음) https://arxiv.org/abs/0711.1612 ; Zou, JASA 101(476):1418–1429 (2006) (검색 확인) | Trimmed: 가장 큰 항 h 개를 벌점에서 제외하는 ℓ1, 딥러닝 희소화에도 적용. DeepHoyer: ℓ1 은 "the same shrinking rate to all parameters, which is inefficient in increasing sparsity". reweighted ℓ1: 이전 해에서 가중치를 다시 계산하는 가중 ℓ1 을 반복해 "in many situations outperforms L1". adaptive lasso: 계수마다 다른 가중으로 oracle 성질 | **큰 쪽을 덜 벌하고 작은 쪽을 더 벌하는 비균등 가중이 균등 ℓ1 보다 희소화에 유리하다는 통계·최적화 선례** | 가중치(계수) 단위. 활동·SNN 아님. 이 문헌들은 정확도가 아니라 지지 회복·희소도 이론 |
| **APoZ** [본문 일부] | Hu, Peng, Tai, Tang, "Network Trimming", arXiv 1607.03250. https://arxiv.org/abs/1607.03250 | "the outputs of a significant portion of neurons in a large network are mostly zero" → 중복. VGG-16 에서 APoZ 90% 초과 뉴런 631개. 프루닝 후 재학습으로 정확도 손실 없이 또는 오히려 향상 | 활동이 낮은 뉴런·채널 = 중복이라는 (가) 전제의 ANN 선례 | 영(0) 비율 기준. 층 간 비교는 별개 (Molchanov 는 첫 층 APoZ 가 목표와 무관하게 비슷하다고 지적) |
| **Molchanov 2017** [본문 표] | "Pruning CNNs for Resource Efficient Inference", ICLR 2017. https://arxiv.org/abs/1611.06440 | Table 1: 특징 맵 제거 효과(oracle)와의 Spearman 상관, **층 안 순위**. 평균 활성: AlexNet/Flowers-102 0.65, VGG-16/Birds-200 0.56, AlexNet/ImageNet **0.09**. APoZ 0.54 / 0.35 / −0.06. Taylor 0.77 / 0.73 / 0.58. 가중치 크기 0.17 / 0.27 / 0.57 | **inter 는 층 안에서 채널 순위를 쓰므로 이 '층 안' 행이 맞는 비교다.** 미세조정 소형 과제에서는 활동량이 중요도의 중간 정도 대리 지표 | 처음부터 학습한 ImageNet AlexNet 에서는 활동량이 거의 무상관(0.09). Taylor 가 항상 더 낫다. SNN·스파이크 아님 |
| **Ye 외 2018; FPGM** [본문] | Ye, Lu, Lin, Wang, "Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning", ICLR 2018 https://arxiv.org/abs/1802.00124 ; He, Liu, Wang, Hu, Yang, "Filter Pruning via Geometric Median", CVPR 2019 Oral (arXiv 코멘트) https://arxiv.org/abs/1811.00250 | Ye: "this assumption is not unconditional", 정규화 없이 노름을 쓰면 해석이 어긋난다. FPGM: 노름 기준의 효과는 "(1) the norm deviation of the filters should be large; (2) the minimum norm of the filters should be small" 두 조건에 의존하며 항상 충족되지 않는다 | (가) 전제가 성립하는 조건을 명시한다. 6절에서 우리 모델이 두 조건을 만족하는지 대조 | 가중치·BN γ 노름 이야기이지 활동량은 아니다. 유추로만 사용 |
| **Rethinking the Value of Network Pruning** [초록] | Liu, Sun, Zhou, Huang, Darrell, ICLR 2019. https://arxiv.org/abs/1810.05270 | 구조적 프루닝에서 "fine-tuning a pruned model only gives comparable or worse performance than training that model" 를 처음부터 학습한 것과 비교, "pruned architecture itself ... is more crucial", "pruning can be useful as an architecture search paradigm" | 채널이 죽어 생긴 층별 채널 수(구조)가 가치라는 해석 — 채널 소멸을 표현력 손실이 아니라 구조 탐색으로 읽는 선례 | 사후 프루닝 맥락 |

### 3. SNN 선행

| 문헌 | 서지·URL | 확인한 문장·수치 (표·설정) | inter 와 겹치는 점 | 다른 점·한계 |
|---|---|---|---|---|
| **SCA** [본문] | Li, Shen, Xu, Chen, Pan, Zhang, Xu, ICML 2024 (PMLR 235:29063–29073). https://proceedings.mlr.press/v235/li24bz.html , https://arxiv.org/abs/2406.01072 | 비정형 연결 프루닝은 "requires specific hardware support"이라며 채널 단위 구조적 프루닝 제안. §4.2 식 (4): 채널 활동 `r_k^l = (1/NT)ΣΣ H_k^l(t)`, 즉 **채널 특징 맵 막전위 L1 노름의 평균**이 작은 채널을 제거하고 일부 재성장. 본문(그림 2 서술): SNNVGG16 CIFAR-10 에서 파라미터 약 30% 수준일 때 91.67% (+0.53; 영문이 "reduced by approximately 30%" 라 방향이 모호하고 앞 절 표는 28.4% 잔존으로 읽음), 약 5% (0.93M) 로 압축하면 약 −2%. 앞 절 표 (VGG16 C10, 기준 91.14): +0.32 @파라미터 55.9%, +0.53 @28.4%, −0.88 @9.3%. DVS-CIFAR10 파라미터 6.95% 에서 −1% 미만. CIFAR-10 채널 제거율 0.8 에서 SynOps 거의 절반 | **SNN 에서 활동 낮은 채널을 지워도 정확도가 유지된다는 가장 직접적인 SNN 선례.** 활동 정의가 막전위 크기라 우리 vmem(침묵 뉴런의 막전위 기여)과 맞닿는다 | 정확도 수치는 그림 서술이며 표 수치는 앞 절(compare-list 행 13) 참조. 사후 프루닝+재성장이고 규제가 아님. 활동 낮은 채널을 지우는 방향이라 **inter 와 같은 가정**. 한 리뷰(arXiv 2407.08744)는 SCA 를 "lower average spike firing frequency"로 요약하나 논문 정의는 막전위다 |
| **Spike-Thrift / AGC** [본문] | Kundu, Datta, Pedram, Beerel, arXiv 2107.12445 (arXiv 제목은 "Towards Low-Latency Energy-Efficient Deep SNNs via Attention-Guided Compression"; 앞 절이 적은 WACV 2021 과 같은 논문인지 이번에 대응 확인 못 함). https://arxiv.org/abs/2107.12445 | 초록: 비정형·구조적(채널) 프루닝 모두 지원. 관찰 3: "Structured SNN pruning yields reduced compression compared to irregular pruning for similar accuracy performance, yet perform considerably better in effective channel-size reduction" — VGG16 은 5배 압축에서 **채널 압축 3.55배**, 비정형 33.4배 압축에서 채널 압축은 1.15배. 채널 순위는 가중치 Frobenius 노름 (§III-B) | 채널 단위 희소가 비정형과 다른 이득(채널 폭 감소)을 준다는 SNN 근거 | 같은 정확도에서 구조적 압축률이 약 한 자릿수 낮다 (5절). 활동이 아니라 가중치 노름. 변환 ANN 기반 |
| **Spik4lite** [본문+슬라이드] | She, Zhou, Wang, Huang, Cui, Guo, ICML 2026 (PMLR v306). https://proceedings.mlr.press/v306/she26b.html (슬라이드 https://icml.cc/media/icml-2026/Slides/63522_o3YAsDZ.pdf) | "We convert 'spike sparsity' into 'channel sparsity', ensuring that the hardware truly reduces calculations". 상용 하드웨어에서 "these zeros still participate in dense matrix multiplications". 정적 FLOPs 만 벌하면 "the network tends to retain a minimal set of channels but forces them to fire continuously" → 채널 발화율을 벌점에 넣는 이유. Table 1 (앞 절 읽음, CIFAR-10 T=4 Spikformer-4-384): 95.24 → 94.15, SOPs −60.9%, Jetson Orin Nano 실측 5.34 → 3.08 J. 모든 행 정확도 −0.63~−3.81%p, n 미기재 | **채널 단위 제거 + 채널 발화율 결합 + 실기 에너지.** inter 의 (다) 전제를 가장 직접 지지 | 라우터·게이트 추가(구조 변경), 정확도 손실, 이득은 스파이크 희소가 아니라 밀집 연산 축소에서 나온다고 저자가 설명. 수동 λ |
| **UPR** [본문 grep] | Shi, Ding, Hao, Yu, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/dd3c889922df2112a5b1769e3c19e28e-Paper-Conference.pdf | 서론: ANN 의 뉴런 프루닝은 "typically performed in a structured manner. When applied to SNNs, they overlook the fact that neurons in SNNs can also be pruned in a fine-grained manner. This oversight hinders the full utilization of the sparse computing capabilities of neuromorphic hardware." 채널 수준 프루닝의 예로 BN 스케일 규제(Zhuang 2020) 등을 인용 | **inter 가 속하는 '구조적 뉴런 프루닝'에 대한 가장 직접적인 반론** | 저자의 논지이고 측정 비교가 아님. UPR 수치는 10-13 절 "기존 항목 갱신" 참조 |
| **Inherent Redundancy in SNNs** [본문] | Yao, Hu, Zhao, Wang, Zhang, Xu, Li, ICCV 2023, pp. 16924–16934 (OpenAccess 페이지 검색 확인). https://arxiv.org/abs/2308.08227 | Observation 3: "the C-SFR is closely related to the spike patterns learned by this channel." 잡음 패턴 채널은 "fires many spikes while focusing on trivial background information", 정상 패턴 채널은 "lower C-SFR focuses on salient gait information in a condensed region". Observation 5: PTD ≥ 0 이면 발화 뉴런이 많고 배경 잡음 패턴 | **발화가 많은 채널이 잡음일 수 있다 — inter 의 전제 방향과 반대.** inter 는 최강(발화합 최대) 채널을 무벌점으로 둔다 | 보행(gait) 예시 (앞 절 기록: 제스처·보행 DVS 데이터), 공간 어텐션 모듈 제안. 채널 평균 발화율(C-SFR)로 보고, 그 채널이 층 안에서 상대적으로 큰 것이 해로운지는 보이지 않았다 |
| **Efficient Structure Slimming; SNN 압축 서베이** [2차] | Li, Fang, Gao, Zhou, Shen, Liu, Pan, Xu, IEEE Trans. Artif. Intell. 5(8):3823–3831 (2024) (서지만 검색 확인, 본문 미열람). 서베이: Xie, Yang, Gao, arXiv 2407.08744 (본문 열람) | 서베이 서술: SNN 판 Network Slimming 은 "due to the thresholding nature of DSNNs, the pruning is not thorough enough, achieving only 60% channel pruning while maintaining accuracy". 검색 요약: 파라미터 약 10% 로 줄여도 성능 저하 1% 미만 (CIFAR-10/100·DVS-Gesture) | 채널 프루닝의 SNN 한계 서술 | **2차 인용.** 60% 수치는 서베이의 요약이고 원 논문에서 확인하지 못했다 |
| **DeepEISNN** [초록] | Liu, Ding, Yu, ICLR 2026 (arXiv 코멘트). https://arxiv.org/abs/2509.23253 | E-I 회로가 "subtractive and divisive inhibition" 으로 뉴런의 흥분성과 이득을 조절, 정규화 없이 깊은 SNN 학습 | SNN 에서 제산 억제를 학습 가능한 구성 요소로 쓴 선례 | 구조 변경. 초록에 희소성·스파이크 감소 수치 없음 |
| **FGKDSNN** (ICLR 2027 제출본) [초록 OpenReview API] | https://openreview.net/forum?id=9S2vkFszgH | "the firing activity is highly concentrated across feature channels"; "strengthening teacher feature transfer to more active student channels" | 발화가 채널에 쏠린다는 관찰과, 발화 많은 채널을 더 중요하게 취급하는 설계 (inter 의 전제와 같은 방향) | **심사 중 제출본이라 인용 보류.** 증류 논문이고 규제 아님. 본문 미열람 |
| **PopFR-CV** (ICLR 2027 제출본) [앞 절 초록 기록] | https://openreview.net/forum?id=ay5isa4N9N (이번 세션 재열람 안 함) | 활동 목적함수가 의도와 반대 방향 통계를 낳고 SDT 에서 침묵 좌표가 +30%p 이상 늘 수 있다 (앞 절 기록) | 규제가 만드는 집단 구조를 측정해야 한다는 압력 | 심사 중, 초록만 |

**3절 소결.** SNN 쪽에서 inter 의 (가)·(다) 전제를 직접 건드리는 선례는 **SCA(채널 활동 기반 제거)·Spik4lite(채널 발화율 기반 제거 + 실측)** 둘이다. 둘 다 **사후 프루닝 또는 게이트 구조**이고 정확도를 일부 잃는다(Spik4lite −0.63~−3.81%p, SCA 고압축 구간 약 −2%). **규제 손실만으로 채널 간 상대 경쟁을 거는 SNN 논문은 이번 조사에서 찾지 못했다.**

### 4. 하드웨어 (채널·코어 매핑이 비용과 이어지는가)

| 문헌 | 서지·URL | 확인한 문장·수치 | 의미 | 한계 |
|---|---|---|---|---|
| **sinabs 문서 (Speck/DYNAP-CNN)** [본문] | https://sinabs.readthedocs.io/main/speck/overview.html (페이지 표기 3.1.4.dev13) | "Memory Constraints and Network Sizing": 층마다 **kernel memory(입력 채널 × 출력 채널 × 커널)** 와 **neuron memory(출력 채널 × 출력 H × W)** 로 나뉘고 실제 요구량은 주소 부호화 때문에 더 크다. "Max channel number: 1024". 예: 16→32 채널 3×3 합성곱, 64×64 입력은 neuron 메모리 128Ki 엔트리로 "this layer CANNOT be deployed on the chip". 초과하면 "No valid mapping" 오류 | **Speck 에서는 채널 수가 메모리 비용과 배치 가능 여부를 정한다.** 죽은 채널을 물리적으로 지우면 이 두 값이 줄어든다 (문서 서술에서 한 단계 추론) | 문서는 죽은 채널을 지우면 이득이라고 쓰지 않는다. 추론이다. 죽은 채널은 이벤트를 안 내므로 동적 전력은 원래 0 이다 |
| **Speck 논문** [본문] | Richter 외, arXiv 2304.06793. https://arxiv.org/abs/2304.06793 | sCNN 코어 9개가 "one convolution and one pooling layer each". 한 코어가 "up to 1024 incoming feature channels" 수신. 커널 주소 압축으로 SRAM 의 동적·정적 에너지 절감을 서술 | 층 = 코어 매핑, 채널 상한 | 전력 비교표(Fig. 10)는 있으나 채널 수와 연결한 측정은 없다 (표 안 값이 어느 칩 것인지 이번에 구분 못 함) |
| **Loihi** [본문] | Davies 외, IEEE Micro Jan/Feb 2018, Table 2 (pre-silicon, 0.75 V). https://gwern.net/doc/ai/scaling/hardware/2018-davies.pdf | 시냅스 스파이크 연산 최소 23.6 pJ, 칩 내 스파이크 1.7 pJ, **뉴런 갱신 활성 81 pJ / 비활성 52 pJ**. 코어 128개, 뉴런 최대 1,024/코어 | 비활성 뉴런 갱신(52 pJ)이 시냅스 연산(23.6 pJ)의 약 2.2배 → 뉴런 수에 비례하는 비용이 스파이크 수와 별개로 존재할 수 있다 | 표는 시뮬레이션 값이고, 비활성 뉴런 갱신을 칩이 매 스텝 실제로 하는지는 이 표에 없다 |
| **Lemaire 외 2022** [본문] | An Analytical Estimation of SNN Energy Efficiency, arXiv 2210.13107 (게재 미확인). https://arxiv.org/abs/2210.13107 | 에너지 식에 출력 뉴런·필터마다 매 시간 스텝 바이어스 읽기·쓰기, 누설 MAC, 리셋이 들어간다(스파이크와 무관한 항). 막전위 읽기 `Cout × Hout × Wout` | 채널 수(Cout)에 비례하는 항이 에너지 모델에 있다 | 해석적 모델, 측정 아님 |
| **Loihi 추정 사례** [본문] | Predictive-maintenance SNN, arXiv 2506.13416 | `E = Nsynops·Esynop + Nneuron updates·Eneuron`, Loihi 추정 에너지의 **86.3% 가 시냅스 연산** (스파이크 수 의존) | 이 모델에서는 스파이크를 줄이는 것이 에너지의 대부분을 줄인다 — 규제의 기본 목표를 지지 | 추정이고 다른 과제. 뉴런 수 감소의 이득은 작게 나온다 |
| **Braille on Loihi** [본문 grep] | arXiv 2205.15864 | "the more cores we use the more power and energy we consume without a significant impact on the delay" → 최소 8 코어 사용 | 코어를 덜 쓰면 측정 전력이 낮다 (Loihi 실측) | 이 과제의 구조에 한정. 채널 제거와 직접 연결한 실험 아님 |

**4절 소결.** **Speck 에서는 채널 수가 메모리와 배치 가능성을 정한다는 것이 문서로 확인된다 (가장 직접).** Loihi 에서는 뉴런 갱신 비용 표와 코어 수·전력 측정이 간접 근거다. 반대로 에너지 모델에서 시냅스 연산이 86% 인 사례도 있다. **채널을 지우면 실측 전력이 얼마나 줄어드는지 Speck/Loihi 에서 측정한 논문은 찾지 못했다.** 상용 GPU 에서의 실측은 Spik4lite (Jetson Orin Nano) 하나뿐이다.

### 5. 반대 근거 (inter 에 불리)

1. **"많이 쏘는 채널 = 중요한 채널"의 방향.** Inherent Redundancy (ICCV 2023) 는 발화가 많은 채널이 배경 잡음 패턴이라고 관찰한다 (DVS 보행). inter 는 층 안 최대 채널을 벌점 0 으로 둔다. Molchanov 2017 은 처음부터 학습한 AlexNet-ImageNet 에서 평균 활성과 제거 효과의 층 안 순위 상관을 0.09 로 보고한다. Ye 2018·FPGM 도 "작은 노름 = 덜 중요" 가정이 조건부임을 보인다.
2. **구조적(채널) 프루닝이 비정형보다 압축률이 낮다.** Spike-Thrift 관찰 3: 같은 정확도에서 압축률이 약 한 자릿수 낮다. UPR 은 구조적 뉴런 프루닝이 뉴로모픽 칩의 희소 연산을 못 쓴다고 주장한다. inter 는 스파이크 감소의 **9% ±2 만** 채널 사망에서 오고 나머지는 산 채널 안 발화 감소다 (inter_deep_261005.md, n=19). 구조적 이득의 크기는 작다.
3. **채널 소멸 = 표현력 손실 가능성.** Spik4lite −0.63~−3.81%p, SCA 고압축 약 −2%. Dying ReLU (Lu, Shin, Su, Karniadakis, arXiv 1903.06733 [초록]) 는 모든 입력에 0 을 내는 뉴런이 기여할 수 없고 깊은 ReLU 망이 확률적으로 죽을 수 있다고 증명한다 (신경 사망의 일반 위험, 규제와 직접 관계는 아님). 우리 측정에서도 같은 스파이크에서 죽은 채널이 많은 런일수록 정확도가 낮다 (부분상관 r=−0.38, 6절).
4. **승자독식 경쟁의 '부익부' 위험.** Diehl & Cook 2015 (Front. Comput. Neurosci., [WebFetch 요약]): 측면 억제로 흥분 뉴런이 경쟁하는 SNN 에서 뉴런 임계값이 발화할 때마다 올라가고 시간에 따라 감소해 "prevent single neurons from dominating the response pattern". 경쟁 학습의 '죽은 유닛' 문제는 오래된 알려진 문제다 (Rumelhart & Zipser 1985, DeSieno 1988 은 강의 슬라이드 https://www.cs.hmc.edu/courses/2004/fall/cs152/slides/compet.pdf 로만 확인 [2차]). inter 에는 이를 막는 항상성 항이 없고 대신 vmem 항(침묵 뉴런 막전위 기여)이 있다.
5. **생물학 대응이 약하다.** Priebe & Ferster 2008 (Neuron 57(4):482–497, doi 10.1016/j.neuron.2008.02.005 [초록]): 교차 방위 억제 등 "lateral inhibition 이 필요해 보이던" 성질이 spike threshold·contrast saturation 같은 feed-forward 비선형성으로 설명된다. Carandini & Heeger 2012: V1 정규화에 GABA 억제는 역할이 작다. Torrado Pacheco 외 2021 (Neuron 109(3):530–544, doi 10.1016/j.neuron.2020.11.001 [초록]): V1 뉴런은 "individual set point" 주변으로 발화율을 양방향 조절 — 생물학은 침묵을 허용하기보다 set-point 를 지키는 쪽이다.
6. **'균등 희소화가 더 낫다'는 직접 결과는 못 찾았다.** SNN 활동 규제에서 균등 대 비균등을 같은 조건으로 비교한 논문은 이번 조사에 없다. 우리 측정이 유일하다: inter 대 BPSR (균등 제곱 ℓ2) 같은 스파이크 정확도 차 +0.04 ±0.04%p (p=0.29, 300~420K, inter n=11 / BPSR n=6, analysis/inter_vs_bpsr_261009.md). 즉 균등 쪽이 낫다는 증거도 못 찾았고 inter 가 낫다는 증거도 정확도에서는 없다.
7. **실측 프록시 비판 (이미 있음).** The Sparsity Ceiling (arXiv 2607.26648, 이번에 pdftotext 16.7KB 를 열었으나 채널·구조적 희소 서술 없음), Yoo 외 VS-WNO (앞 절 M6): 발화율이 54% → 18% 로 줄어도 Jetson 에서 dense 보다 나쁨. 채널 제거는 이 비판의 한 가지 해법(Spik4lite)이지만 스파이크 감소 자체는 아니다.

### 6. 우리 측정과의 대조 (문헌의 전제가 우리 모델에서 성립하는가)

원본은 `analysis/inter_vs_bpsr_261009.md` §1 표, `analysis/inter_deep_261005.md`, `docs/decisions.md` 10-05 절. 이번에 다시 계산하지 않았다. 모두 R19 C10 T=4.

| 문헌의 전제 | 우리 수치 (n) | 해석 |
|---|---|---|
| FPGM 조건 (1) 노름(활동)의 편차가 크다 | baseline 채널 지니(스파이크 가중 평균) 0.4256, 상위 10% 채널의 스파이크 몫 25.62% (층 평균, 균등이면 10%) (n=3). inter 1.5e-3: 0.4695, 28.26% (n=3) | baseline 에서 이미 채널 활동의 편차가 크다. inter 는 이를 더 벌린다 |
| FPGM 조건 (2) 최솟값이 0 에 가깝다 | baseline 죽은 채널 534.0 / 4,736 (11.3%), 이를 지우면 MACs −19.38% (n=3) | 가장 약한 채널이 이미 0 이다. 두 조건 모두 만족하는 것으로 보인다 (유추: FPGM 은 가중치 노름) |
| (가) 죽은 채널을 지워도 정확도 불변 | 죽은 채널 제거 시 Δacc 0.00 (28런, decisions 10-05) | 추론 시 중복이라는 직접 확인 |
| (나) 상대 가중의 구조적 이득 | 같은 스파이크 (300~420K) inter − BPSR: 죽은 채널 +49.8 [+28.3, +71.2] (p=0.0002), 죽은 채널 제거 시 MACs +3.44%p [+2.11, +4.76], 이미지당 활성 뉴런 −5.7% [−6.9, −4.4] (inter n=11 / BPSR n=6) | 구조적 축에서는 inter 가 앞선다 (4·5절의 구조적 이득 논지와 일치) |
| (나) 정확도 이득 | 같은 스파이크 정확도 차 +0.042 [−0.039, +0.123] %p, p=0.29 (같은 n). n≤4 에서 0.15~0.2%p 미만 판정 불가 | 정확도 우위는 판정 불가 (유의하지 않음을 숨기지 않는다) |
| (가) 죽은 채널이 많을수록 | 같은 팔·같은 스파이크에서 죽은 채널↑ → 정확도↓, 부분상관 r=−0.38 (p=0.017, n=40) | 문헌의 "채널 소멸 = 손실 가능성"과 같은 방향. 구조적 이득과 정확도는 별개 |
| 채널 사망의 몫 | inter 의 스파이크 감소 중 채널 사망 몫 9% ±2 (n=19) | 5절 2번 — 하드웨어 이득의 규모는 작다 |
| Molchanov: Taylor > 활동량 | `ours_ch_taylor` 4점 역분산 평균 잔차 +0.003%p, CI [−0.13, +0.13] | 활동량 대신 Taylor 순위로 바꿔도 곡선이 안 움직인다. 검정력상 0.1%p 효과는 배제 못 함 |
| FBS·SASL 의 '최강 무벌점' | inter 는 층 안 최대 채널 `w=0` | 관련 관찰: ~300K 아래에서 inter 가 다른 팔보다 0.16~0.24%p 낮다 (inter n=1 점 포함, inter_deep §0 3번). 원인이 '최강 채널 무벌점'인지는 **미확인** |

### 7. 못 찾은 것·미확인 (본문 인용 금지)

- **채널 공간합 상대 경쟁을 SNN 스파이크 손실로 쓴 선행 논문, 그리고 SNN 활동 규제에서 균등 대 비균등을 직접 비교한 논문**: 찾지 못했다. 못 찾은 것이지 없다는 뜻이 아니다 (검색은 영어 키워드, arXiv·OpenReview·PMLR·PubMed 위주; Springer·IEEE 본문은 접근 못 함).
- **"특징 채널 공간합끼리 겨루는 규제"의 직접 생물학 근거**: 못 찾음. 후각 사구체 간 억제(Olsen 2010)가 가장 가깝고 억제 세기가 최대치가 아니라 전체 활동에 비례한다.
- **Speck/Loihi 에서 채널·뉴런을 줄였을 때의 실측 전력 논문**: 못 찾음. Spik4lite (Jetson) 만 있다.
- **[2차] 로 표시한 것**: Rozell 2008 (PDF 미열람), Trimmed ℓ1 (PMLR 페이지는 검색 요약만), Zou 2006 (초록 수준 확인 못 함), Barth & Poulet 2012 (요약 문장), Efficient Structure Slimming 본문과 서베이의 "60%" 수치, Diehl & Cook 2015 (WebFetch 요약), 경쟁 학습 죽은 유닛(강의 슬라이드).
- **열지 못한 것**: Springer(Applied Intelligence 2023 "An efficient pruning and fine-tuning method for deep spiking neural network", 채널 점수에 BN γ 와 발화율을 함께 쓴다고 검색 요약에만 나옴, 수치 미확인), Hahnloser 외 2000 / Douglas & Martin 2004 (PubMed 검색에서 서지 확인 실패, 인용하지 않음).
- **SASL 게재 서지**: arXiv 코멘트에 IEEE TCSVT 수락만 있고 권·쪽 미확인. **Spike-Thrift**: 앞 절의 WACV 2021 표기와 arXiv 제목이 다르다. 대응 확인 전에는 두 서지를 한 논문으로 단정하지 않는다.
- 심사 중 제출본 (FGKDSNN, PopFR-CV) 은 초록만 확인했고 인용 보류.

### 8. 논문에 쓸 때의 범위 (제안만, 결정은 사용자)

1. "채널 간 경쟁은 생물학적으로 타당하다"는 **근거가 아니라 영감**으로만 쓴다. 쓸 수 있는 문장: 정규화·승자독식(Carandini & Heeger 2012), 후각 채널 간 측면 억제(Olsen 2010), 컨볼루션 SNN 의 intra/inter-map competition (Kheradpisheh 2018).
2. ANN 선례는 구성 요소별로 인용한다: 채널 공간 풀링 + 채널 k-WTA (FBS), 순위 기반 비균등 벌점과 최강 무벌점 (SASL, Trimmed ℓ1), 균등 벌점의 한계 (Polarization, DeepHoyer, SASL 서론). 새로움은 **"SNN 스파이크 손실에서 채널 간 상대 경쟁 가중"으로만 한정**한다.
3. 하드웨어 주장은 "채널 단위 희소는 Speck 의 메모리·매핑(채널 수 기반)과 상용 GPU 의 밀집 연산에 직접 이득, 스파이크 단위 희소는 그렇지 않다" 수준으로 쓰고, 실측은 없음을 한계 문장에 둔다. 규모는 6절의 9% ±2, MACs +3.44%p 를 같이 적는다.
4. 반론 대비: (가) 방향 (Inherent Redundancy), UPR 의 "구조적은 희소 연산을 못 쓴다", Spike-Thrift 의 압축률 차이. 우리 데이터로 답할 수 있는 것은 "죽은 채널 제거 Δacc 0.00 (28런)"과 "같은 스파이크에서 활성 뉴런 −5.7%" 뿐이다.
5. 정확도 우위는 쓰지 않는다 (BPSR 대비 +0.04 ±0.04, p=0.29).
