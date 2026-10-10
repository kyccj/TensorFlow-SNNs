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
