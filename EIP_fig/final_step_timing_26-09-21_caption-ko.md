# 그림 4 (단일폭) — 규제 시점(final_step) 캡션 (국문)

`final_step_timing_ko_26-09-21.{pdf,svg,png}` (영문 정본: `final_step_timing_en_26-09-21.*`)

## 코드 근거 (neurons.py:784-792, 1081-1090)

```python
# 784-792
if conf.reg_spike_out and not (conf.reg_spike_final_step and t != conf.time_step):
    ...
# 1081-1090
if conf.reg_spike_final_step:
    spike_reg = tf.add_n([self.out.read(_i) for _i in range(conf.time_step)])
else:
    spike_reg = spike
```

- **plain L2** (`reg_spike_final_step=False`): 조건문이 매 `t` 마다 참이 되어, 규제가
  타임스텝마다 걸린다. 이때 `spike_reg` 는 그 순간의 `spike`(순간값)이고, `sc_rate` 는
  "그 시점까지의" `spike_count` 로 계산된다. `t=1` 에서는 어떤 뉴런도 최대 1번만 쏠 수
  있으므로 그룹의 max 도 1 — `sc_norm`·`sc_rate` 가 0 또는 1 두 값으로만 무너진다
  (0/1 이분법).
- **ours** (`reg_spike_final_step=True`): 조건문이 `t=T` 에서만 참이 되어, 규제가 한
  번만 걸린다. `spike_reg` 는 `t=1..T` 의 `self.out`(TensorArray, 미분 가능)을 전부
  더한 값 — 시간축 전체를 적분한 뒤 규제하는 것과 같다. `sc_rate` 도 최종
  `spike_count`(0..T 정수 전 구간)로 계산되어 0/1 이분법이 생기지 않는다.

## 그림이 보이는 것

T=4 타임라인 위에 두 방식이 규제를 거는 시점을 점으로 표시했다.
- **plain L2**: 4개 점(t=1,2,3,4) 전부가 규제 발생 시점이다. `t=1` 에 0/1 이분법 문제를
  화살표로 짚었다.
- **ours**: 마지막 점(t=4)만 채워져 있다(규제 발생 시점). 앞의 세 점은 빈 원으로,
  "규제는 안 걸리지만 spike 는 여기서도 누적된다"를 나타낸다. 괄호 표시는 `t=1..T` 를
  적분해 `t=4` 의 규제로 들어간다는 것을 보인다.

## 이 그림에 없는 것

w¹ vs w² 기울기 경로 차이(`ours` vs `ours_w1`, 곱셈이 `custom_gradient` 안/밖에 있는지)
는 별도 그림(그림 5, `w1_vs_w2_gradient_fig.py`)에서 다룬다. 두 그림은 서로 다른
축(그림 4=언제 규제가 걸리는가, 그림 5=역전파 경로가 어떻게 다른가)을 다루므로
분리했다.

## 지킨 것
- 실험 수치 없음, T=4·이산값(0,0.25,0.5,0.75)은 illustrative
- 색맹 안전 팔레트(파랑/주황) + 선종류(속 찬 점 vs 빈 원) 병행
