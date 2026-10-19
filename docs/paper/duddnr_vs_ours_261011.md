# VGGSNN CIFAR10-DVS: 2025-11 duddnr 코드(81~82%) vs TensorFlow-SNNs(79.7~80.3%) 차이 (26-10-11)

비교 대상
- 11월: scratchpad/duddnr_full/ (파일 날짜 2025-11-27 ~ 12-05). 이하 `N:`. **주의(10-11 11:00): 81~82% 런의 실제 실행 코드는 세종 `~/PycharmProjects/Base` 이며 입력 처리가 N 과 다르다 (표 #0).**
- 우리: /home/kyccj/PycharmProjects/TensorFlow-SNNs, 실제 config 는 _paper/vggdvscm-base_sg-asym-w1p0-h0p75-s1/config_sweep.py. 이하 `O:`
- 제외(이미 반영): 증강 nda_cutmix, asy_height_fix 이식.
- 표시: [확인] = 코드로 확인함, [추측] = 코드 밖 추정.

## 차이 목록 (영향 큰 순)

| # | 차이 | 11월 값 | 우리 값 | 예상 영향 | 확인 방법 |
|---|---|---|---|---|---|
| **0** | **입력 프로토콜 (10-11 11:00 가설 → 10-12 22:10 기각: 프레임 0 런 80.14 / 11월 복제 런 78.83)** | Base/Surro-release: model.py 가 5차원 입력을 통째로 쓰고 InputGenLayer 가 `inputs[:,0]` 반환 → **매 스텝 프레임 0** | 스텝 t 에 프레임 t. 단 BN 시간평균 때문에 **4프레임 평균 하나**와 같음 (측정: 평균 반복 = 일반 80.34 동일) | **~1.5~2%p** (11월 80.2~82.4 n≈13 vs 우리 79.7~80.3). 11월 1~5번 항목은 부차적 | 검증 런 `_ours/vggdvscm-base-f0-s1` (데이터 수준 프레임0 반복) 1런. 논문 프로토콜 선택은 사용자 결정 (decisions 10-11 11:00) |
| 1 | 서로게이트: 적응형 폭 + 높이 | `adaptive_surrogate=True`, SGV+TGC 둘 다 켬 (N:config_snn_training_surro.py:100-116). 폭 β 를 층×스텝마다 [0.1, 1.0] 의 100개 후보 중에서 500 iter 마다 다시 고름 (N:neurons.py:2055, 2139-2172). 식은 h=(v−1)/(2β)+**0.5** (높이 0.5 고정, N:neurons.py:1759, 2094-2095) | 고정 폭. 10-10 4런은 높이 0.75 / 0.6 (O:flags.py:1083 기본 0.75, O:neurons.py:2298-2304). **h0.5 와 적응형은 한 번도 돌리지 않음** | 클 수 있음 [추측]. 우리 고정 서로게이트 4종만으로도 78.8~80.3 (각 n=1), 폭 1.5%p | ① `ahf-w1.0-h0.5` 1런 (β=1 로 고정한 11월 식과 같음). ② 적응형 β 가 결정적이면 SGV/TGC 이식 |
| 2 | 검증셋에 무작위 좌우 반전 | `as_frames` 가 증강 인자와 상관없이 매번 `random_flip_left_right` 를 건다 (N:datasets/events/image.py:165). 검증도 이 함수를 씀 (N:cifar10_dvs.py:182) | 반전은 `augmentation=True` 일 때만 (O:image.py:173-186). 검증은 결정적 | **없음 (측정)**. 추측은 +0.3~1%p 였으나 우리 런도 마지막 40에폭 val SD 가 1.1~1.6%p 이고 best 가 평균보다 약 2~3%p 높음 (아래 표). 반전이 에폭마다 독립 잡음을 더하면 최댓값이 더 오른다 | **측정 완료(10-11 09:40, 기각)**: best ckpt(80.34) 를 반전 검증 20회 → 평균 79.81, SD 0.49, 최대 80.75. 반전은 평균을 0.5%p 내리고 잡음만 더하며, 200에폭 최댓값 모의에서도 부풀지 않는다 (decisions 10-11 09:40) |
| 3 | conv1 뒤 BN 개수 | 2회: Conv2D(use_bn=True) 안의 BN(eps 1.001e-5, tdBN 켬) → bn_conv1(eps 1e-4, tdBN 끔) (N:models/vggsnn.py:112-114, N:lib_snn/layers.py:118-132) | 1회 (O:models/vggsnn.py:72-87) | 작음~중간 [추측]. tdBN 배율은 n_init_vth=1.0 이라 효과 없음 (O:layers_new/batch_normalization.py:232). 남는 것은 BN 재정규화와 γ/β 한 쌍 추가(가중치 감쇠 대상) | vggsnn 에 플래그 하나로 이중 BN 복원 후 1런 |
| 4 | 실행 환경 | 서버·TF·tfa·tfds 빌드 미상. 홈이 /home/duddnr254 인 다른 서버 | canus TF 2.12.1. tfa 는 TF≥2.13 용이라 경고 뜸 (train.log). tfds cifar10_dvs 0.0.3 은 2026-03-28 빌드 | 미상. tfds 빌드가 다르면 `train[:10%]` 검증 구성이 다를 수 있음 [추측] | duddnr 서버에서 ~/tfds_val_sig.py 를 돌려 ~/tfds_val_sig.json(1000개 label/n/t0/t1/tsum)과 대조 |
| 5 | 런 수와 선택 편향 | 81~82 는 n=1~2 로 추정 (런 기록 미발견) | baseline 79.74 ±0.17 (n=3) + 서로게이트 4종 각 n=1 | DVS 시드 SD ~0.8 (decisions). n=1 의 81 은 우리 분포의 위쪽 꼬리와 겹칠 수 있음 | 11월 텐서보드에서 런 수·시드 확인 |

같음으로 확인한 항목은 아래 축별 상세에 [같음]으로 적었다.

### 우리 런의 검증 정확도 궤적 (train.log, 200에폭, 138)

| 런 | best (에폭) | 마지막 40에폭 평균 | SD | 최소 | 마지막 |
|---|---|---|---|---|---|
| vggdvscm base s1 | 79.64 (181) | 77.48 | 1.11 | 75.40 | 78.83 |
| vggdvscm base s2 | 79.94 (187) | 78.08 | 1.20 | 73.89 | 79.44 |
| vggdvscm base s3 | 79.64 (199) | 77.39 | 1.64 | 72.78 | 78.83 |
| base_sg asym w1.0/h0.75 s1 | 80.34 (199) | 77.67 | 1.43 | 75.10 | 79.03 |
| base_sg ahf w1.0/h0.75 s1 | 78.83 (183) | 76.19 | 1.28 | 72.98 | 77.52 |

학습률이 1e-5 까지 내려간 끝 구간에서도 val 이 ±1.5%p 흔들린다. best 는 대략 "평균 + 2σ" 짜리 통계다.

## 축별 상세

### 1. 데이터
- tfds 분할: 둘 다 `train[10%:]`(shuffle_files=True) / `train[:10%]`. 표본 단위 `.shuffle()` 없음. [같음] (N:cifar10_dvs.py:48-49, O:47-48)
- 프레임: T=4. dt=(t_end−t_start)//T, 마지막 칸은 T−1 로 접음. 극성 2채널, 이벤트당 0.1 을 더함 (scatter_add). [같음] (N:image.py:17-18, 114-149 / O:image.py 같은 식)
- resize 48×48: 학습 nearest, 검증 bilinear. 둘 다 같은 불일치. [같음] (N:image.py:163, 244 / O:image.py:170, 296)
- 학습 증강: flip → batch(drop_remainder) → nda_cutmix. N:cifar10_dvs.py:86-171 과 O:146-237 은 한 줄씩 같다. 유일한 차이는 λ~Beta(1,1)(tfp) 대 U(0,1) 인데 같은 분포다. [같음]
- **검증: 11월만 무작위 좌우 반전.** [다름, 표 #2]
- 라벨 one-hot, 10클래스. [같음]
- 에폭 정의: 9000//32 = 281 step, 검증 1000//32 = 31 배치 = 992장. 둘 다 `train_ds.cardinality()`. [같음] (N:datasets.py:104)
- randaug_en / rand_erase_en: 11월 config 에서 True 이고 3월 ckpt 경로명에 `_ra_re` 가 붙지만, DVS 경로에서 읽는 곳이 없다. 11월 코드에서 grep 해도 image_cls 계열에만 있다. [같음: 무시됨]

### 2. 모델
- 채널 64-128-AP-256-256-AP-512-512-AP-512-512-AP-FC(10), avg pooling, dropout 0, glorot_uniform, bias 있음. 출력은 n_fc(VMEM) 다음 softmax. [같음]
- **conv1 이중 BN.** [다름, 표 #3]
- 첫 층 tdBN 은 POISSON 입력일 때만, 나머지 층은 tdBN 켬(배율 vth=1). [같음]
- 모델 출력은 마지막 스텝 (N:model.py 3582 `read(T-1)` / O:model.py:3540 `last_ts`). [같음]

### 3. 뉴런
- LIF: v_t = 0.9·(v_{t−1} + I_t). v ≥ vth 면 발화하고 v −= spike (reset by subtraction, vth=1.0 고정). 리셋 경로에 gradient 차단 없음. [같음]
  (함수별 diff: leak / integration / run_type_hid / run_type_out / run_type_in 차이 0줄. fire_rate 는 dtype 와 reset_to_zero 분기만 다름)
- 입력층은 REAL 을 그대로 통과. [같음]
- 서로게이트 창은 둘 다 ≥ / ≤ (닫힌 구간). [같음]
- **서로게이트 식·폭·적응.** [다름, 표 #1] 11월 β 기록 두 벌 (Surro/VSNN_beta_tgc_history, 디렉토리 날짜 2026-03-29라 11월 런과 같은 런인지는 미확인):
  - `asy_height_fix_ours`: 8층 × t0~t2 전 에폭 β=1.00. decisions.md:375 의 근거가 이것이다.
  - `asy_height_fix_sgv+tgc_ours`: β 가 움직인다. 에폭 1-20 / 21-50 / 51-100 / 101-150 / 151-200 구간 평균:
    - t0: 0.51 / 0.79 / 0.93 / 0.96 / 0.99
    - t1: 0.62 / 0.64 / 0.72 / 0.83 / 0.96
    - t2: 0.82 / 0.83 / 0.88 / 0.93 / 0.98
  - t3(4번째 스텝, SGV 가 갱신)은 기록되지 않았다.
  - 스냅샷 config 는 SGV+TGC 둘 다 켜고, 저장 경로가 `ablation/ours_reproduce/1/` (N:config:153) 이다. 즉 81~82 는 Surro **제안법** 런일 가능성이 높다.
- 높이 출처: 11월 스냅샷 0.5 (상수). Surro 저장소 2025-11-17 커밋 f437cc8 은 asy_height_val=0.6 ("VGG16 : 0.6 ResNet19 : 0.75"). 우리 기본 0.75 는 ResNet19 값이다.
- 수식 비교 (v ∈ [0, 2]):
  - 11월 β=1: h = 0.5v, 0 → 1, 넓이 1.0
  - 우리 ahf w1.0/h0.75: h = 0.5v + 0.25, 0.25 → 1.25, 넓이 1.5
  - 우리 asym w1.0/h0.75: h = 0.25v + 0.625, 0.625 → 1.125
- spike dtype float32, mixed precision 꺼짐. 우리 build 의 `vmem_init` 만 fp16 상수(O:neurons.py build)인데 값이 0.0 이라 영향 없음.
- leak_const_train=False, n_type=LIF, im_en / rmp_en / reg_psp_SEL / adaptive_vth_SEL 은 flags 기본 False 이고 각 항은 `if conf.*` 로 막혀 있다 (N:neurons.py:776, 790, 818, 833). VSNN_DVS 블록의 값들은 쓰이지 않는다. [같음]

### 4. 학습
- 최적화기: `tf.keras.optimizers.Adam(learning_rate, weight_decay=2e-2)`. β·eps 기본값(0.9 / 0.999 / 1e-7). `exclude_from_weight_decay` 를 양쪽 다 호출하지 않아 BN γ/β 와 bias 까지 감쇠한다. [같음] (O:model_builder.py:172, N 같은 줄)
- lr: lib_snn.optimizers.CosineDecay (파일 diff 0). 1e-5 → 6e-3 를 20에폭 워밍업, 이후 코사인, 하한 1e-5. [같음] (O:dvs_warmup_epochs=20)
- 배치 32, 200에폭, label smoothing 0.1, grad clip 없음, TF32 꺼짐 (config.py:280, tf32_mode=False). [같음]
- 손실: CE(label_smoothing) on softmax(마지막 스텝 VMEM). train step 은 temporal-first 단일 테이프 (N:_compute_gradients, O:tape.gradient, 같은 값). [같음]
- 정확도 지표 categorical_accuracy, 검증은 매 에폭. best = ModelCheckpoint(monitor val_acc).best. [같음]
- 우리 config 의 `reg_spike_log_detail=True` 는 기록용 변수 갱신뿐이고 손실에 들어가지 않는다 (reg_spike_out=False). [영향 없음]

### 5. 기타
- 시드: 11월은 시드를 설정하지 않음 (main 에 없음). 우리는 run_seed=-1 이라 손대지 않음. [같음]
- flags 기본값 중 바뀐 것: n_init_vth 0.5 → 1.0 (두 config 모두 1.0 으로 덮음), en_record_output True → False (기록용), debug_surro_grad_per_iter. [영향 없음]
- data_aug_mix 'nda' 가 11월 flags enum 에 없지만, absl 속성 대입은 enum 검증을 거치지 않아 그대로 통과한다. [확인]

### 6. 결과 집계
- 두 코드의 best_val_acc = ModelCheckpointResume.best (lib_snn/callbacks.py:119-126). 992장 검증셋 val_acc 의 200에폭 최댓값이고, 텐서보드 스칼라 'best_val_acc' 로도 기록된다. [정의 같음]
- 단 11월 val_acc 는 무작위 반전 아래에서 잰 값이라, 같은 정의라도 분포가 더 넓고 최댓값이 더 높다 (표 #2).
- 11월 텐서보드·런 디렉토리는 이 서버에서 찾지 못했다 (/media/hdd1/kyccj/ICML26/YC/ 는 2026-03 rebuttal 런).
