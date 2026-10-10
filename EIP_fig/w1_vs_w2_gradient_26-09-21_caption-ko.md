# 그림 5 (단일폭) — w¹ vs w² 기울기 경로 캡션 (국문)

`w1_vs_w2_gradient_ko_26-09-21.{pdf,svg,png}` (영문 정본: `w1_vs_w2_gradient_en_26-09-21.*`)

## 코드 근거 (layers.py:2021-2093)

두 경로는 **forward 가 완전히 같다**: `out = sqrt(sum((spike·sc_rate)²))`. 차이는
곱셈 `x = spike·sc_rate` 가 `tf.custom_gradient` 로 감싼 상자 **안**이냐 **밖**이냐
하나뿐이다.

- **`ours` (w², 기본값, `reg_spike_wta_rev_w1=False`)**: `neurons.py:1091` 에서
  `sc_loss = spike_reg * sc_rate` 로 곱셈이 **상자 밖**에서 일어나고, 그 결과 `x` 를
  `l2_norm_wta_rev(x, sc_rate, name)` 에 넘긴다. 상자 안 backward 는 `dL/dx =
  sc_rate/‖x‖` 를 돌려주는데, 이건 **x 에 대한** 기울기다. 상자를 나가면 autodiff 가
  `x=spike·sc_rate` 의 곱셈을 다시 미분해 `dx/dspike=sc_rate` 를 한 번 더 곱한다.
  결과: `∂L/∂spike = sc_rate²/‖x‖` (코드 docstring: "실측 확인").
- **`ours_w1` (w¹, `reg_spike_wta_rev_w1=True`)**: `l2_norm_wta_rev_w1(spike, sc_rate,
  name)` 이 `spike` 를 직접 받아 `x = spike·sc_rate` 를 **상자 안에서** 계산한다.
  상자가 돌려주는 기울기가 곧 `spike` 에 대한 기울기이므로 autodiff 가 `sc_rate` 를
  다시 곱하지 않는다. 결과: `∂L/∂spike = sc_rate/‖x‖` (한 번만 곱함).

`sc_rate` 는 `spike_count`(`tf.Variable`)에서 나와 기울기가 없으므로 두 함수 모두
`sc_rate` 쪽 기울기는 `None` 이다.

## 그림이 보이는 것

- **위 두 패널**: `spike → x → ‖x‖ → loss` 계산 그래프. 점선 상자 = `custom_gradient`
  경계. 실선 화살표 = forward(두 경로 동일). 점선 화살표 = backward, 주황 점 =
  `sc_rate` 가 곱해지는 지점. **`ours`(w²) 는 주황 점이 두 개**(상자 안 한 번 + 상자를
  나간 뒤 autodiff 로 한 번 더) — **`ours_w1`(w¹) 은 한 개뿐**이다.
- **아래 패널**: `sc_rate ∈ [0,1]` 에서 `w` 대 `w²` 곡선(illustrative). 발화 뉴런은
  `sc_rate<1` 이므로 `w² < w` — 같은 `ρ`(loss-ratio)에서 w¹ 경로가 발화 뉴런에
  더 센 압력을 준다. 이 방향은 실측(같은 ρ 에서 `ours_w1` 이 스파이크를 25% 더
  줄임)과 일치한다. **수치는 그림에 없다 — 곡선의 모양만 그렸다.**

## 반드시 읽어야 할 결론 (2026-09-21 확정)

1. forward 가 같으므로 loss-ratio 의 `R`(=spike_reg 크기)도 같고, **ρ 눈금이 두 팔
   사이에서 공유된다**.
2. w² 는 의도된 설계가 아니라 **곱셈 위치에서 생긴 부산물**이다. `ours_w1` 은
   "설계 의도대로 w 한 번이면 무엇이 달라지는가"를 묻는 팔이다.
3. w² 경로는 `w → c·w` 스케일에 불변이다(`λ→λ/c, w²→c²w², R→cR` 이 상쇄). w¹ 은
   불변이 아니다.
4. **두 경로는 정확도-스파이크 곡선, 뉴런 분포, 채널 수 전부에서 구별되지 않는다.**
   w¹ 이 바꾸는 것은 사실상 `ρ` 눈금뿐이다. **이 그림은 "무엇이 다른가"를 보이는
   것이지 "무엇이 더 낫다"가 아니다** — 결론 문장은 그림이 아니라 이 캡션과 본문에
   쓴다.

## 그림 4 와의 관계

그림 4(규제 시점, `final_step`)는 "언제" 규제가 걸리는지를 다루고, 이 그림은
"역전파 경로가 어떻게 다른지"를 다룬다. 원래 그림 4 캡션에 넣기로 했던 내용을
사용자 요청으로 별도 그림으로 분리했다.

## 지킨 것
- 실험 수치(스파이크 25% 감소 등) 는 이 캡션에만 있다 — 그림에는 없다
- 곡선(w vs w²)은 illustrative 로 명시
- 색맹 안전 팔레트(파랑=backward, 주황=sc_rate 곱셈 지점, 청록/자홍=w¹/w² 곡선)
