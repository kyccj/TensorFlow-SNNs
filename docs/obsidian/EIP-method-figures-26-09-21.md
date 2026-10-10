---
tags: [EIP, SNN, Method, 그림]
날짜: 2026-09-21
---

# EIP Method 절 개념도 — 2026-09-21

전부 illustrative(예시 숫자, 실험 결과 아님). 영문 정본은 저장소 `EIP_fig/*_en_26-09-21.*`
(pdf/svg/png), 여기 임베드된 건 검토용 국문본(`*_ko_26-09-21.png`)이다. 스크립트는
`EIP_fig/*_fig.py` (`--lang en|ko` 로 재생성 가능).

> [!warning] 볼트 복사는 사용자가 한다
> 나스(`161.122.23.163`)에 접속하지 않았다. 이 노트와 `attachments/` 의 PNG 5장은
> 저장소 `docs/obsidian/` 안에 있다. 나스를 켠 뒤 아래 복사 명령으로 볼트에 반영할 것:
> ```
> cp docs/obsidian/EIP-method-figures-26-09-21.md /home/kyccj/00.obsi/<대상폴더>/
> cp docs/obsidian/attachments/*_26-09-21.png /home/kyccj/00.obsi/<대상폴더>/attachments/
> ```

## 그림 1 — 경쟁 범위 3종 (`ours_intra_ch` / `ours_layer` / `ours_inter_ch`)

2026-09-21 사용자 피드백 반영: 남색 칸 글자색 자동 반전(휘도 기준), 이름 교체
(`within_channel→ours_intra_ch`, `none→ours_layer`, `channel→ours_inter_ch`),
채널 간 격차를 키운 예시 맵 재설계 + `ours_layer` 패널에 `ours_intra_ch` 대비
차이 칸 표시(△, n=8/48).

![[competition_scope_ko_26-09-21.png]]

캡션: `EIP_fig/competition_scope_26-09-21_caption-ko.md`

## 그림 2 — 가중치 함수 sc_rate(sc_norm)

2026-09-21 수정: 열린 원(○)=공유점(여러 곡선이 겹침) / 채운 원(●)=고유점(그 곡선만
도달)의 의미를 범례에 명시.

![[weight_function_ko_26-09-21.png]]

캡션: `EIP_fig/weight_function_26-09-21_caption-ko.md`

## 그림 3 — vmem readiness

`spike_count` 가 0..T=4 정수라 침묵 뉴런(층의 78~97%, 실측·그림에는 미기재)이 전부
`sc_rate=1.0` 에 뭉치는 문제와, `readiness=clip(vmem/vth,0,1)` 로 침묵 뉴런만
`[0,gain)` 에 퍼뜨리는 해법.

![[vmem_readiness_ko_26-09-21.png]]

캡션: `EIP_fig/vmem_readiness_26-09-21_caption-ko.md`

## 그림 4 — 규제 시점(final_step)

T=4 에서 `plain L2` 는 매 타임스텝 규제(→ t=1 에서 0/1 이분법), `ours` 는 `t=T` 한
번만(누적 spike 합 사용).

![[final_step_timing_ko_26-09-21.png]]

캡션: `EIP_fig/final_step_timing_26-09-21_caption-ko.md`

## 그림 5 — w¹ vs w² 기울기 경로 (`ours` vs `ours_w1`)

곱셈 `x=spike·sc_rate` 이 `custom_gradient` 상자 안(w¹, 한 번 곱함)이냐 밖(w²,
두 번 곱함)이냐. **결론: 두 경로는 정확도-스파이크 곡선·뉴런 분포·채널 수에서
구별되지 않는다** — 이 그림은 차이를 보이는 것이지 우열을 주장하지 않는다.

![[w1_vs_w2_gradient_ko_26-09-21.png]]

캡션: `EIP_fig/w1_vs_w2_gradient_26-09-21_caption-ko.md`
