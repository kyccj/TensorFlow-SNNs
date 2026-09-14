# 표·그림 인벤토리 (Task 5, 2026-09-14)

정본은 `docs/superpowers/specs/2026-09-14-eip-paper-spec.md` §8 이다. 이 문서는 그 번호를
그대로 옮기고, 8쪽 제약 아래 본문/부록 배치와 그 근거만 정한다. **번호를 새로 만들지 않는다.**
Task 6·11·12·13 은 이 문서의 번호로만 표/그림을 부른다.

본문 최소 구성 (스펙이 못 박음): **T1 · T3 · T4 · F1 · F4 · (F2+F3, 2패널 합본)**.
나머지는 스펙 §8 이 제안한 배치를 기본값으로 하되, 8쪽 여유를 봐서 T2 → T8 순으로 본문에
올리고 안 되면 부록으로 내린다.

## 표 (T1–T8)

| ID | 내용 | 생성 명령 | 게이트 | 배치 | 근거 |
|---|---|---|---|---|---|
| T1 | R19-C10 주 비교표 — base/l2/sm/prop 이 190~202K 스파이크 구간에 모임 | `python collect_paper.py '_paper/r19c10-{base,l2,sm,prop}-*' --out _results/t1_r19c10_main.csv` 후 baseline 은 스펙 §2 A1 정본 5런 목록(`_paper/r19c10-base---s11`, `---s12`, `_baseline_more/base_r19_c10_run3`, `_baseline_no_reg_r19_c10`, `_baseline_trajectory`)으로 수동 대체 — 글롭만 쓰면 2런만 잡혀 평균이 어긋난다 | 없음 | **본문 (필수)** | 헤드라인 C1(무손실 51.5% 감소)의 유일한 직접 증거. 스펙이 본문으로 못 박음 |
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
| F1 | 정확도–스파이크 곡선, 가중 설계 19종이 한 곡선에 | **스크립트 없음 — Task 13 에서 작성** | 없음 | **본문 (필수)** | 스펙이 본문으로 못 박음. A8(19종 등가성)의 유일한 시각적 증거이자 P4 의 핵심 관찰 |
| F2 | 층별 지니 계수 (baseline/l2/prop) | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 c) | 없음 | **본문 (필수, F3 과 2패널 합본)** | 스펙이 F2+F3 합본을 본문 최소 구성으로 못 박음 |
| F3 | 층별 상위 10% 점유 | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 d) | 없음 | **본문 (필수, F2 와 2패널 합본)** | 위와 동일 |
| F4 | 같은 예산에서 쓰는 뉴런 수 — 회귀 두 선이 전 구간 평행 | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 a) | 없음 | **본문 (필수)** | 스펙이 본문으로 못 박음. A2(활성 뉴런 13.1% 차이) 의 직접 증거, CONTRIB-1 의 핵심 그림 |
| F5 | 뉴런당 발화 강도 막대 (4방법) | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python neuron_usage.py` (패널 b) | 없음 | 부록 | A3(뉴런당 발화 강도 순서) 의 보조 증거 — 본문 서사(F1·F4·F2+F3)가 이미 "곡선 동일 / 뉴런 사용 차이" 를 전달하므로, 강도 축은 부록에서 수치로만 보강 |
| F6 | 에폭별 λ 궤적과 정확도 (ep200 cutmix 종료 반응) | `cd EIP_fig && /home/kyccj/anaconda3/envs/venv_1/bin/python lambda_epoch.py` | **§9-B — ep201 급락이 LR 스케줄과 겹치는지 미확인** | 부록 | 본문은 T1·T2·T3·T4·F1·F4·(F2+F3) 만으로 이미 8쪽이 찬다. F6 은 CONTRIB-2(G4 대조군 미완, 조건부 서술)를 뒷받침하는 그림이라 8쪽 안에서 우선순위가 가장 낮다 — 본문 지면 예산이 근거이지, 궤적을 보이는 것 자체가 어떤 주장을 함축해서가 아니다. **캡션 의무**: F6 을 쓰는 절은 ep201 λ 급락이 CutMix 종료 시점과 겹친다는 것과, 그 원인이 LR 스케줄과 분리 확인되지 않았다는 것을 반드시 명시한다. **재평가 조건**: §9-B(LR 스케줄과의 겹침 확인)가 해소되면 F6 은 본문 승격 후보로 재검토한다 — 지금의 부록 배치는 최종 결정이 아니다 |
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
