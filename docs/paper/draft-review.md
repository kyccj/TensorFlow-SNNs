# 전체 자체 검토 (Task 15, 2026-09-14)

**전제.** 이 머신에는 LaTeX 엔진(`pdflatex`/`bibtex`/`xelatex`/`lualatex`/`tectonic`/`latexmk`)이
전혀 없다. `make -C paper`·`pdfinfo`를 쓸 수 없어 **실제 컴파일은 수행하지 않았다.**
대신 `python paper/texcheck.py paper`(구조 검사 — 괄호/환경 균형, `\cite` 키, `\input`
대상 존재)와 수동 대조를 썼다. **아래 결과는 컴파일 검증이 아니다.** TeX Live 설치 후
`make -C paper`로 반드시 재검증할 것.

```
$ python paper/texcheck.py paper
검사한 .tex 14개, refs.bib 키 28개, 문제 0건
```

---

## 1. 수치 대조

`paper/sections/*.tex`·`paper/tables/*.tex`에 등장하는 숫자 값을 전수 추출해
`docs/superpowers/specs/2026-09-14-eip-paper-spec.md` §2(A/B/C)·§9·§12-3 및
`docs/paper/claim-evidence.md`와 대조했다. **확인한 고유 수치는 약 90개**이고,
표(T1/T3/T4/T8)의 개별 런 행·평균 행은 spec §2 A1/A1b/A6(B3)/B2 표와 자릿수까지
전부 일치하며, 평균 행의 산술(스파이크 합/4, 정확도 합/4 등)도 직접 재계산해
확인했다(예: baseline 481,809+488,900+484,578+499,190 → 488,619.25 → "488,619" 일치).

**지시받은 구식 수치(486,656 / 96.654 / 51.5% / 96.524 / "19 designs") 재검사 —
0건 발견.** `grep`으로 전 섹션·전 표를 훑었고 하나도 남아 있지 않다. "13종" 표기도
`related.tex`에 정확히 반영되어 있다(spec §2 A8/`figures-tables.md` "13종" 재검증 결과와 일치).

**추적 불가 / 상충 2건:**

1. **[중간] `discussion.tex:44-45`의 λ 급락 수치가 최신 재실행 로그와 다르다.**
   본문: "It drops 44% at epoch 201, from $4.75\times10^{-7}$ to $2.68\times10^{-7}$"
   — 이는 spec §9-B의 서술을 그대로 옮긴 것이다. 그러나 `docs/paper/figures-tables.md`
   "그림 스크립트 실행 확인(2026-09-14, fix round 1로 재실행)" 절의 **재현 가능한**
   `lambda_epoch.py` 실행 로그는 `ep200 4.71e-07  ep201 2.97e-07`로, 하락폭이
   44%가 아니라 약 37%다. 같은 날짜(2026-09-14)에 갱신된 두 소스가 다른 값을 보고하고
   있고, F4/F2F3처럼 "이 숫자가 정본"이라는 명시적 판정이 λ 수치에는 없다. §9-B 자체가
   이 급락 원인이 미확인이라 명시하므로 본문 주장에는 영향이 없지만, **수치 자체는 spec
   본문(§9-B, 옛 값)과 spec의 부속 문서(figures-tables.md, 재실행 값)가 불일치한다** —
   Task 15는 게이트가 아니라 정정 권한이 없으므로 여기서는 발견만 보고한다.
2. **[경미] `discussion.tex:64`의 "window width $\alpha=0.5$"는 스펙 원문에 수치가 없다.**
   spec §9-E는 "`surro_grad_alpha` 값도 함께 적는다"고만 지시할 뿐 실제 값을 주지 않는다.
   본문의 0.5는 코드베이스에서 직접 가져왔을 것으로 보이나, **스펙 문서만 근거로 삼는다는
   스펙 0항의 규칙("여기 없는 수치는 논문에 쓰지 않는다")으로는 이 리뷰에서 검증할 수 없다.**

**부수 관찰(추적 불가는 아님) — T4 표 내부 산술.** `t4_cost.tex`의 퍼센트 열은
spec §2 A7 표를 자릿수까지 그대로 옮겼지만, 표시된 ms/step 값으로 직접 재계산하면
"제안법"(695 vs 631) = 10.14%, 1-softmax(759 vs 631) = 20.29%, final-step 제거(841 vs 631)
= 33.28%로, 표기된 10.0%/20.2%/33.2%와 소수 첫째 자리에서 어긋난다. **이 오차는 spec §2
A7 표에 이미 있던 것을 그대로 옮긴 것이라 이번 초안이 새로 만든 문제는 아니다** — spec이
더 정밀한(반올림 전) ms/step 값에서 퍼센트를 계산했을 가능성이 높다. 보고만 하고 수정은
컨트롤러 판단에 맡긴다.

---

## 2. 인용 대조

```
refs.bib 키: 28개 / 본문 \cite 키: 21개 (고유)
사용됐지만 refs.bib에 없는 키: 0건
refs.bib에 있지만 한 번도 인용 안 된 키: 7건
  krizhevsky2009learning, li2017cifar10dvs, orchard2015converting,
  yao2023spikedriven, yao2024metaspikeformer, yao2025scaling, zhou2023spikformer
```

미인용 7개는 전부 §3 Method(현재 공백)·T2(SOTA 표, 아직 없음)·CIFAR-10/100 데이터셋
자체 인용·Spikformer/SDT-V3 관련 문헌으로, 아직 채워지지 않은 절이 쓸 근거로 보인다
(아래 §8 참조). **`meda2022improving` 단독 인용 — 0건, 확인됨.** 두 등장 모두
`neil2016learning, sorbaro2020optimizing, pellegrini2021low`와 함께 묶여 있다
(`related.tex:17`, `intro.tex:31`).

**`\cite`가 아닌 `\ref` 대조에서 발견된 별도 문제** (§5 참조) — `\ref{tab:t8_generality}`
(`experiments.tex:193`)와 `\ref{fig:f5_intensity}`(`experiments.tex:325`)가 살아있는
본문에서 쓰이는데, 두 `\label`이 정의된 `paper/tables/t8_generality.tex`·
`paper/figures/f5_intensity.tex`는 어느 쪽도 `\input`되지 않는다(각각 게이트 주석으로
빠짐). `pdflatex`가 있었다면 이 두 곳은 "??"로 찍히고 `Reference ... undefined` 경고가
났을 자리다 — texcheck.py는 `\ref`를 검사하지 않아 잡지 못했다.

---

## 3. 기각 주장 검사

Task 7 grep을 영문 대응 패턴으로 확장해 `paper/sections/*.tex`에 적용:

```
grep -nE 'WTA (happen|occur|emerge)|accuracy advantage|throughout the network|
          across the (whole|entire) network' paper/sections/*.tex
```

4건 히트 — `abstract.tex:17`, `experiments.tex:95,154,290`. **전부 부정문 안에서
등장한다** ("without an accuracy advantage", "we do not assert it as an accuracy
advantage", "neither result depends on claiming an accuracy advantage", "we do not
describe this as concentration 'throughout the network'"). §2.6/§5의 "일어나지 않았다"
서술과 같은 성격의 예외이므로 **위반 0건.** 한국어 패턴(`정확도.*우위` 등)은 영문 본문
특성상 해당 사항 없음.

---

## 4. Method 누설 검사

`maxnorm|final_step|final-step|vmem|loss-ratio|wta_rev|sc_rate|abl_no|run_paper\.py|
reg_detail\.csv|neuron_usage\.py|collect_paper\.py`를 전 섹션에 검사한 결과 히트는
`related.tex:51`의 `% GATE: G4 -- [내부용, 본문 반입 금지] ...` 주석 한 줄뿐이며,
이는 브리프가 명시한 면제 대상이다. `t3_ablation.tex`의 행 이름("w/o potential",
"w/o inversion", "w/o final-step")은 `method-terms.md` 오른쪽 열 그대로이므로 면제.
§5(discussion.tex)의 "sub-threshold term"·"the ratio controller"·"loss-ratio" 언급은
spec §9-C를 갚는 자리이므로 면제.

**그러나 브리프가 명시한 대로 수식·파일 경로·레이어 이름·텐서 모양은 위 세 면제 중
어느 것으로도 면제되지 않는다.** 이 기준으로 별도 검사한 결과, **3건의 파일 경로 노출을
찾았다:**

| 위치 | 내용 | 심각도 |
|---|---|---|
| `intro.tex:57` | `\texttt{python extract\_neuron\_spikes.py --split test --n 10000}` — G5 게이트 문단 안에서 CLI 명령 전체가 렌더링될 본문에 그대로 노출 | **중간** — 소개(§1) 산문에 스크립트 파일명·플래그가 직접 등장. `[내부용, 본문 반입 금지]` 태그가 없고 세 면제 항목 어디에도 해당하지 않는다 |
| `discussion.tex:123` | `\texttt{run\_paper.py}` — Limitations 불릿 "training configuration files that `run_paper.py` points to..." | **경미** — 재현성 한계(spec §12-6)를 정직하게 고지하는 문장의 일부이고 스크립트 이름 자체가 그 한계의 핵심 대상이라 맥락상 불가피에 가깝지만, 브리프 기준으로는 파일 경로 노출이라 목록에 남긴다 |
| `discussion.tex:124` | `\texttt{.gitignore}` — 같은 불릿 안 | **경미** — 위와 동일 문장, 동일 사유 |

`experiments.tex`의 `\texttt{dead\_neuron\_ratio}`(4회), `\texttt{asym}`,
`\texttt{run\_seed}`, `\texttt{tf.data}`는 Method(§3, 규제항 자체의 내부 구조)가
아니라 **실험 설정·데이터 수집 절차**를 가리키는 용어라서 이번 검사의 "Method 누설"
범주 밖으로 판단했다 — 다만 `run_seed`는 코드 설정 플래그 이름이 그대로 노출된
사례라 경계선에 있다는 점은 적어 둔다.

---

## 5. 게이트 대조

spec §2/§8/§12가 정의한 게이트 목록(G1–G8, G1은 09-14 해소) 대 본문 `% GATE:` 마커:

| 게이트 | 마커 위치 | 상태 |
|---|---|---|
| G1 | (해소, 마커 불필요) | 해소됨, 본문에 마커 없음 — 정상 |
| G2 | `discussion.tex:33` | 있음, 문장 헤지됨 |
| G3 | `experiments.tex:182` (G6·G8과 합본) | 있음, 문단 전체가 헤지됨 |
| G4 | `intro.tex:73`, `related.tex:51`, `conclusion.tex:8` | 있음(3곳), 모두 헤지됨 |
| G5 | `intro.tex:52` | 있음, 플레이스홀더(`__G5_SILENT_RATIO__`)로 값 자체가 비어 있음이 명시됨 |
| G6 | `experiments.tex:182` | G3와 동일 위치 |
| G7 | `intro.tex:81`, `related.tex:66` | 있음(2곳), Speck 문장은 커밋 처리(주석)되어 아예 렌더링되지 않음 |
| G8 | `experiments.tex:182` | G3와 동일 위치 |

**목록에 있는데 마커가 없는 게이트 — 0건. 마커가 있는데 목록에 없는 게이트 이름 — 0건**
(`t3_ablation.tex:37`의 `% GATE: T3-matched`는 spec §12-4 tier B 승인 대기를 가리키는
표 내부 라벨로, G1–G8 번호 체계 밖이지만 spec 자체가 정의한 대상이라 "존재하지 않는
게이트"로 보지 않는다).

**게이트 마커는 있으나 마커가 지시한 조치가 실행되지 않은 1건 — [중간]**
`related.tex:51`의 G4 주석은 "G4가 미충족 상태로 남으면 이 서브섹션 전체를 §2에서
Discussion으로 옮겨라"고 명시한다. 2026-09-14 기준 G4는 여전히 미충족(`claim-evidence.md`,
`figures-tables.md` 전부 확인)인데, `\subsection{Balancing auxiliary loss terms}`는
여전히 §2(Related Work)에 그대로 있다. 문장 자체는 "provisionally/조건부로"라는 헤지가
있어 과장된 주장으로 드러나진 않았지만, **주석이 스스로 요구한 위치 이동은 실행되지
않았다.**

**게이트된 문장이 무헤지 단정으로 흘렀는가 — 발견 0건.** G2/G3·G6·G8/G4/G5/G7 문단
모두 "we do not X" 형태로 닫혀 있다.

---

## 6. §9 고지 의무 대조

| 항목 | 이행 여부 | 위치 |
|---|---|---|
| A. wta_rev는 실제 도함수가 아니다 | 이행 | `discussion.tex:23-31` (용어 노출 없이 "designed, monotone update rule, not the derivative" 로 정확히 서술) |
| B. λ 급락 원인 미확인 | 이행 (단, §1 수치 상충은 §6 참조) | `discussion.tex:44-51` |
| C. loss-ratio 궤적/강도 혼재 | 이행 | `discussion.tex:33-42` (G2 문단) |
| D. SynOps 축 제외 | 이행 | `discussion.tex:53-59` |
| E. 대리 기울기 코드베이스 차이 | 이행 | `discussion.tex:61-73` |
| F. QKFormer 포팅 버그 | 이행 | `discussion.tex:75-78` (spec 초안 문구 그대로) |
| G. eval 프로토콜 통일 | 이행 | `discussion.tex:80-88`, `experiments.tex` Setup "Spike counting" 문단, F2/F3/F4 캡션 |
| H. "이미지당 조용"↔"영구 침묵" 구분 | 이행 | `discussion.tex:90-98`, `experiments.tex` "What silent means here" 문단 |
| I. 재현성 부록 인프라 사고(rsync/tfevents, ls -A) | **미이행 — 누락** | 전 섹션 `grep -inE 'rsync|tfevents|fsck|inode'` 0건 |

**§9-I가 빠져 있다.** 원인은 구조적이다 — `main.tex`에 `\appendix`가 아직 없고
(`experiments.tex:332-334` 주석이 "이 파일은 아직 `\appendix` 절이 없다"고 명시), §9-I는
애초에 "재현성 부록에 적을" 내용으로 지정돼 있어 부록 자체가 없는 지금 넣을 자리가 없다.
**Task 완료 요건이 아니라 부록 미착수의 파생 결과다.**

---

## 7. 분량 추정

**방법:** 컴파일 불가하므로 CVPR 10pt 2단 레이아웃의 경험적 단어당 지면 밀도로
역산했다 — 본문 텍스트(캡션 포함) 약 1,500단어/페이지(2단 합산), figure*(전폭)
1개당 약 0.3–0.4페이지, 단일 컬럼 표는 행 수에 비례해 0.05–0.3페이지로 어림했다.
**이것은 실측이 아니라 근사치이며, `pdflatex` 설치 후 `pdfinfo main.pdf | grep Pages`로
반드시 재검증해야 한다.**

| 절 | 단어 수(주석 제외) | 근사 페이지 |
|---|---:|---:|
| Abstract | 200 | ~0.15 (제목 블록과 공유) |
| §1 Introduction | 698 | ~0.5 |
| §2 Related Work | 484 | ~0.35 |
| §3 Method | 5 (공백) | 0 (미작성, 예산 1.5p) |
| §4 Experiments (본문) | 2,604 | ~1.75 |
| — Figure 2·4 (전폭 2개) | — | ~0.7 |
| — Table 1·3·4 (본문 3개) | — | ~0.3–0.4 |
| §5 Discussion(+Limitations) | 1,085 | ~0.75 |
| §6 Conclusion | 100 | ~0.1 |
| **소계(§3 제외, Method 미작성분)** | | **~4.6–4.7페이지** |
| §3 Method 채워졌을 때(spec §7 예산) | | +1.5 |
| **예상 합계** | | **~6.1–6.2 / 8페이지** |

8쪽 예산에 여유가 있어 보인다(참고문헌·부록은 본문 8쪽 카운트에서 보통 제외).
다만 이 추정은 (a) Method 절 실제 분량이 예산과 얼마나 맞는지, (b) 아직 없는 T2·부록
(T5/T6/T7/F5/F6)이 실제로 얼마나 얹힐지에 크게 좌우된다 — **실측 전에는 참고용으로만
쓸 것.**

---

## 8. 남은 게이트 현황

| 게이트 | 내용 | 상태(2026-09-14) | 본문에서 막는 것 |
|---|---|---|---|
| G2 | 어블레이션 표본 보강(8런) | 미실행 | Discussion B0 문장(18.7–20.3%)이 본문에 있지만 게이트 마커로 감싸짐 |
| G3 | CIFAR-100 조건당 n≥3 | 미충족(n=1×2) | §4 Generality 본문 서술만, T8 표는 주석 처리(미포함) |
| G4 | `l2 + loss-ratio` 대조군 | 미실행 | CONTRIB-2("조건부") 전체, §2.4 서브섹션 위치(§5 참조) |
| G5 | 전체 테스트셋(10,000장) 영구 침묵 뉴런 측정 | 미실행 | §1 헤드라인 두 번째 문장의 플레이스홀더(`__G5_SILENT_RATIO__`) |
| G6 | Spikformer/CIFAR10-DVS 제안법 재실행 | 미실행(구 sweep만) | T8/§4 Generality |
| G7 | Speck 실측 | 미착수(레포에 코드 없음) | CONTRIB-3 전체, 헤드라인 세 번째 문장, T6/F7 | 
| G8 | SDT-V3 173M 실행 | 미착수(포팅만 완료) | T8/§4 Generality |
| — | T2(SOTA 대비 표) | **본문에 존재하지 않음** — spec §8이 "본문" 배치로 지정했으나 파일 자체가 없다 | §1 헤드라인을 문헌 맥락에 놓는 근거 부재. 미인용 7개 bib 항목 중 일부가 이 표를 위해 남아 있는 것으로 보임 |
| — | Appendix(T5/T6/T7/F5/F6) | **`\appendix` 자체가 `main.tex`에 없음** | `\ref{tab:t8_generality}`·`\ref{fig:f5_intensity}` 댕글링, §9-I 누락, "in the appendix" 3회 언급이 갈 곳 없음 |

**독자가 바로 행동할 수 있는 우선순위:** (1) `\ref` 댕글링 2건은 컴파일 전에 반드시
고쳐야 하는 구조적 결함(§2/§5 참조) — Appendix를 만들거나 해당 문장을 조건부로 감싸야
한다. (2) `related.tex`의 G4 서브섹션 위치는 게이트 자신의 지시와 어긋난다(§5).
(3) T2 표 부재는 8쪽 여유(§7)를 감안하면 채울 공간이 있다.

---

## 9. 중복 검사

기계적 문장 분리(주석·LaTeX 명령 제거, 토큰 70% 이상 겹침) 스크립트를
`paper/sections/*.tex` 전체에 실행 — **완전 축자 반복(≥95% 겹침) 0건.** 남은 5쌍은
전부 §1(Introduction)의 예고와 §4(Experiments)의 본 진술 사이 겹침이며, 판정은 다음과 같다:

1. **정당한 구조 (패딩 아님).** "at a matched spike budget, ... 13.1% fewer active
   neurons" — §1은 압축된 예고(수치+결론), §4는 회귀 계수·SE·t·n과 프로토콜 주의사항을
   포함한 전체 증거. 서로 다른 일을 한다.
2. **정당한 구조.** "$96.720\%$ ... $488,619$ spikes" (baseline 수치) — §1은 헤드라인
   문장 안에, §4는 표(Table 1)와 함께 재확인. 논문 서두-본문 관례.
3. **정당한 구조.** "coefficient of $0.869$ ($t=-21.5$, $n=21$)" — 위 1번과 사실상 같은
   쌍, §1/§4 각각에서 반복.
4. **정당한 구조.** 뉴런당 발화율 순서(1.621>1.479>1.370>1.284, $t=10.5$) — §1 요약,
   §4 Figure 5/본문 상세.
5. **정당한 구조 (같은 파일 내부).** `experiments.tex` 안에서 문단 첫 문장("On
   ResNet-19/CIFAR-10, ... fewer active neurons than Plain $L_2$")과 이어지는 회귀
   문장의 겹침 — 주제문+근거문 구조로, 같은 단락 안의 정상적 전개다.

**브리프가 언급한 "이전 라운드의 축자 반복"은 이번 검사에서 재현되지 않았다** — 이미
수정된 것으로 보인다. 남은 겹침은 모두 헤드라인이 서론·결과·결론에서 각기 다른 정밀도로
반복되는 표준 논문 구조이며, 추가 조치가 필요한 패딩은 발견하지 못했다.

---

## 요약

- **컴파일:** 불가(엔진 없음). 구조 검사(texcheck.py)는 통과(0건).
- **수치:** 확인 ~90개, 완전 추적 실패 0건, 상충/미검증 2건(λ 급락 수치, surro_grad_alpha=0.5).
- **인용:** 미사용 키 7건(위반 아님, 미완성 절 대기), 누락 키 0건, meda2022improving 단독 인용 0건.
- **기각 주장:** 위반 0건.
- **Method 누설:** 파일 경로 노출 3건(중간 1, 경미 2). 용어 자체 누설은 0건.
- **게이트:** 목록-마커 불일치 0건. 마커의 지시(G4 서브섹션 재배치)가 실행되지 않은 사례 1건.
- **§9 고지:** A–H 이행, I(인프라 사고) 누락(부록 미착수의 파생).
- **분량:** 근사 6.1–6.2 / 8페이지(추정, 미실측).
- **구조 결함(신규 발견):** `\ref{tab:t8_generality}`·`\ref{fig:f5_intensity}` 댕글링 참조 2건, T2 표 부재 1건.
- **중복:** 축자 반복 0건. 나머지 5쌍 전부 정당한 서론-본문 구조.

---

## Fix Round 1 (2026-09-14) — team-lead 판정에 따른 6건 수정

**1. 댕글링 참조 (Critical) — 수정 완료.** `paper/main.tex`에 `\appendix` + `\input{sections/appendix}`
추가, `paper/sections/appendix.tex` 신설(재현성 서브섹션 + F5 `\input`). F5 는 게이트가 없어
그대로 부록에 넣었다. T8 은 G3/G6/G8 미충족이라 계속 뺀다 — `experiments.tex:193`의
`Table~\ref{tab:t8_generality} in the appendix` 문구 자체를 지우고, 표 없이도 뜻이 통하도록
문단을 다시 썼다("the corresponding table is kept out of both the main body and the
appendix until one opens"). F6 은 애초에 라이브 참조가 없었음을 재확인(주석 안에만 있었음).

**2. T2 부재 (Important) — 수정 완료.** `docs/paper/figures-tables.md`의 T2 행을 "미작성 —
문헌 값 수집 필요"로 갱신하고 "8쪽 여유가 있는 한 유지" 서술과 "T2 → T8 순으로 본문에
올리고"라는 상단 문구를 함께 고쳤다. `.tex` 어디에도 T2 참조가 없음을 재확인(0건). 표를
새로 만들지 않았다.

**3. §9-I 누락 (Important) — 수정 완료.** `paper/sections/appendix.tex`의 "Reproducibility"
서브섹션에 spec §9-I 세 요소(`ls -A`가 `.git`/`.omc`/`__pycache__`를 옮긴 사고와 복구,
`rsync --remove-source-files`가 실행 중인 6런의 tfevents inode 를 지운 사고와 `/proc/<pid>/fd/N`
복구, 첫 확인이 grep 패턴 오류로 "정상"이라 오보한 것)를 그대로 적었다 — 사과 없이, 사실만.

**4. λ 수치 stale (Important) — 수정 완료.** `discussion.tex`를 "36.9%, $4.71\times10^{-7}$
(epoch 200) → $2.97\times10^{-7}$ (epoch 201)"로 교체했다. spec §9-B 항목 자체도 재실행
로그를 박아 넣고 "이 절의 수치가 이제 정본"이라고 명시해, F4 회귀와 같은 원칙("재현 가능한
값이 이긴다")을 적용했다. 재실행 로그는 아래 검증 참조.

**5. related.tex:51 주석 (Important) — 수정 완료.** "G4 미충족 시 Discussion 으로 옮겨라"는
지시를 지우고, "2026-09-14 사용자 결정으로 §2에 조건부 서술로 유지, 이 메모가 그 결정을
기록하며 더 이상 실행 대상이 아니다"로 교체했다. 본문(§2.4 서브섹션 자체)은 건드리지 않았다.

**6. 파일 경로 노출 (Minor) — 수정 완료(intro.tex만).** `intro.tex`의 CLI 명령을
`\footnote{}`로 옮겼다("Measured with \texttt{extract\_neuron\_spikes.py --split test
--n 10000}."). `discussion.tex:123-124`의 `run_paper.py`/`.gitignore`는 판정대로 그대로
둔다 — 재현성 한계를 고지하는 문장의 본체이므로 파일명 노출이 곧 그 고지다.

### 검증

```
$ cd EIP_fig && python lambda_epoch.py
λ: ep1 4.35e-07  ep100 4.85e-07  ep200 4.71e-07  ep201 2.97e-07  ep310 2.58e-07  시간평균 4.00e-07
최종 val_acc: 제안법 96.46 / 고정 3e-7 96.29 / 고정 5e-7 96.39
```

```
$ python paper/texcheck.py paper
검사한 .tex 15개, refs.bib 키 28개, 문제 0건
```

댕글링 `\ref` 재검사 — `main.tex`부터 실제 `\input`되는 파일만 추적해(주석 제외) 모든 `\label`을
모으고 모든 `\ref`를 대조하는 스크립트를 돌렸다:

```
Live (transitively \input) files: figures/f5_intensity.tex, main.tex,
  sections/{abstract,appendix,conclusion,discussion,experiments,intro,method,related}.tex,
  tables/{t1_main,t3_ablation,t4_cost}.tex
Dangling \ref (label not defined in any live/\input file): none
```

**분량 재추정 (부록 추가 반영, 여전히 추정치 — 엔진 없음).** appendix.tex 151 단어 +
단일 컬럼 F5 그림 1개가 늘었다. 나머지 절 단어 수는 사실상 그대로(intro 696, related 484,
experiments 2625, discussion 1088, conclusion 100). 본문+부록 합계 약 5,150단어,
같은 밀도 가정(~1,500단어/페이지) + figure* 2개(~0.7p) + 표 3개(~0.35p) + 부록 F5(~0.2p)로
**~3.4(텍스트) + 0.7 + 0.35 + 0.2 ≈ 4.65페이지**, 여기에 Method 예산 1.5페이지를 더하면
**~6.1–6.2 / 8페이지** — Fix Round 1 이전 추정과 사실상 동일하다(부록이 작아서 영향이 작음).

Commit: (아래 참조)
