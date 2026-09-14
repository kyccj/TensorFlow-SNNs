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
