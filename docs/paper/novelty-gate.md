# 선점 판정 — Activity Pruning (NeurIPS 2025) / Sorbaro 2020

판정 근거는 `docs/superpowers/specs/2026-09-14-eip-paper-spec.md` §4. 이 문서는 그 판정을
논문에서 쓸 형태(기여 문장·헤드라인 두 버전)로 정리한 것이다.

## Step 1: 본문 미확인 두 항목

Activity Pruning (Tong Bu, Xinyu Shi, Zhaofei Yu, NeurIPS 2025) 의 OpenReview 본문은
CAPTCHA 로 막혀 있다. NeurIPS 2025 poster 페이지, ML Anthology, researchr, proceedings.com,
Semantic Scholar, 저자(Tong Bu, Peking University) 관련 페이지를 확인했지만 이 논문의
arXiv 미러나 대체 본문 소스는 없었다 — 모두 초록·서지 정보만 제공하고 본문은 OpenReview
PDF 하나로만 연결된다.

- **(a) 하드웨어 실측이 본문에 있는가**: 미확인 — OpenReview CAPTCHA, arXiv/proceedings
  미러 없음. GitHub 저장소(`github.com/putshua/Activity-Pruning-SNN`)에는 배포 코드나
  전력 측정이 없고 `SOPMonitor` 는 SynOps 시뮬레이션 계산이다. 초록도 "Neuromorphic
  hardware 배포를 용이하게 한다" 는 동기만 언급하고 실측값은 제시하지 않는다.
  **코드·초록 범위에서는 확인되지 않음** — 본문에 있을 가능성을 단정적으로 배제하지 않는다.
- **(b) 가중 방식 간 등가성을 다루는가**: 미확인 — 같은 이유. 초록은 AT-LIF 뉴런 모델과
  역치 적응(threshold adaptation) 하나로 발화율을 억제하는 단일 방법을 설명할 뿐, 여러
  가중 방식(softmax/maxnorm/entropy/z-score 등)을 비교하거나 그들이 같은 곡선에
  수렴한다는 논의는 초록에 없다. **코드·초록 범위에서는 확인되지 않음.**

## 확정 기여

```
CONTRIB-1: 가중 설계 19종이 같은 정확도–스파이크 곡선에 수렴하지만, 곡선 위 같은 자리가
           같은 내부 상태를 뜻하지 않는다 — 같은 예산에서 활성 뉴런 수가 13% 다르다.
           [선점 안 됨. 논문의 무게중심]
CONTRIB-2: 규제 세기를 무차원 비율 하나로 정한다. 절대 목표도 예산에 대한 사전 지식도
           필요 없다. [Sorbaro 2020 의 정적 1/S0², Activity Pruning 의 손으로 박은
           eta/eta2 와 대비. **G4 가 채워져야 성립**]
CONTRIB-3: [실기 검증. **G7 이 비면 통째로 뺀다**]
```

## 선점을 인정할 것 (§2.3 에서 먼저 밝힌다)

- "스파이크 감소는 거의 전부 뉴런 침묵이다" — Activity Pruning (NeurIPS 2025) 이 선점.
  우리는 신규 발견이 아니라 **재확인 + 정량화**로 쓴다.
- "기존 활동 규제에 구조적 한계가 있다" — 같은 논문 초록에 명시됨.

## 헤드라인 2번 문장 — 미결 (G5/G7 대기)

```
HEAD-2a (재확인으로 낮춤):
  선행 연구가 관찰한 활동 가지치기를 정량화하고, 그 위에서 어떤 뉴런을 끄는지가
  방법마다 다르다는 것을 보인다.

HEAD-2b (측정 축을 앞세움):
  같은 스파이크 예산에서 활성 뉴런을 13% 적게 쓴다 — 곡선은 두 방법을 구분하지
  못하지만 뉴런 사용은 구분한다.

판정 조건:
  G5 에서 영구 침묵 뉴런 비율이 크면 → 2a 로도 구조적 프루닝 이득을 말할 수 있다
  작으면 → 2b 만 남는다
```

사용자 결정(2026-09-14): 이 두 문장 중 어느 쪽도 지금 지우지 않는다. G5(테스트셋 뉴런
측정)·G7(Speck) 결과가 나온 뒤에 하나를 고른다.
