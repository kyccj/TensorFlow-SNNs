# 사전학습 체크포인트 파인튜닝 + 동일 조건 무규제 대조 — 선례 조사 (2026-10-10)

목적: SDTv3 실험 (a) 규제 없이 동일 에폭 추가 파인튜닝한 대조 vs (b) 규제를 넣고 파인튜닝한 런.
리뷰어의 "더 학습했으니 당연히 좋다" 공격에 "다른 논문도 같은 방식" 이라고 답할 근거.

읽기 전용 조사. 커밋 없음. 원문을 직접 내려받아 확인한 것은 [원문], 검색·요약 도구만 거친 것은 [요약], 못 확인한 것은 "미확인".

## 먼저 알아야 할 것 (불리한 소식)

1. **SNN 분야에서 "같은 체크포인트 + 같은 에폭, 방법만 뺀 파인튜닝 대조" 를 명시적으로 보고한 논문은 찾지 못했다.**
   가장 가까운 TP-Spikformer (ICLR 2026) 는 *같은 체크포인트(SDT-V3 공식)를 파인튜닝하는 프로토콜* 은 선례지만,
   대조군은 공식 체크포인트 수치이지 추가 파인튜닝한 런이 아니다.
   즉 우리 대조 (a) 는 SNN 에서는 **선례를 따르는 것이 아니라 선례보다 엄격한 것**이다.
   답변 문구는 "SNN 선례를 따랐다" 가 아니라 "ANN/LLM 압축 문헌의 표준 통제를 SNN 에 가져왔다" 로 써야 정직하다.
2. 대조를 둔 ANN/LLM 논문 중에도 **정확히 "dense 를 같은 토큰만큼 계속 학습" 한 것은 Sheared-LLaMA (소규모, 저자 스스로 'preliminary') 와
   INTERLACE (arXiv) 뿐**이다. 나머지는 변형 (같은 예산의 정규화 없는 변형, 랜덤 초기화, 같은 스케줄) 이다.
3. ProSparse 에서는 규제 없는 대조가 규제 런보다 정확도가 높았다 (아래). 우리 대조도 그럴 수 있다는 경고.

---

## A. SNN 분야 (확인된 것 3편, 모두 부분 선례)

### A1. TP-Spikformer: Token Pruned Spiking Transformer
- 저자·소속: Wenjie Wei, Xiaolong Zhou, Malu Zhang 외, UESTC / Shenzhen Loop Area Institute 등 [원문 첫 페이지]
- 학회·연도: ICLR 2026 ("Published as a conference paper at ICLR 2026") [원문]
- URL: https://arxiv.org/abs/2603.00527 (PDF https://arxiv.org/pdf/2603.00527)
- 한 일: SDT-V1, QKFormer, **SDT-V3** 의 *공식 사전학습 가중치* 에 토큰 가지치기를 적용하고 파인튜닝.
  원문: "During fine-tuning, we remove the warm-up epoch and set the learning rate to 1e-5 for 50 epochs." (부록, ImageNet-1K) [원문]
  원문: "we prune three architectures using official pre-trained weights, comparing results with and without fine-tuning." (5.2) [원문]
- 대조: 표 2 의 비교 기준은 공식 수치 (예: SDT-V3-19M ×0.78 → 79.01 (-0.71), 표 2). **같은 에폭 무규제 파인튜닝 대조는 없다** [원문 grep 으로 확인, 해당 문구 없음].
- 우리와의 대응: 같은 모델군 (SDT-V3), 같은 출발점 (공식 ckpt), 짧은 저학습률 파인튜닝 → **프로토콜 선례**. 대조는 우리가 더 엄격.
- 불리한 점: 이 논문이 통과했다는 사실은 "대조 없이도 ICLR 에서 받아줬다" 일 뿐, "대조를 두는 것이 관행" 의 근거가 아니다.

### A2. Toward Efficient Spiking Transformers: Synapse Pruning Meets Synergistic Learning-Based Compensation
- 저자: Hongze Sun 외 (UESTC 소속으로 검색됨) [요약, 저자 전원·소속 미확인]
- 학회·연도: arXiv 2508.01992 (2025). 게재 학회 **미확인**
- URL: https://arxiv.org/abs/2508.01992
- 한 일: "our method are directly applied to state-of-the-art (SOTA) pre-trained models" 후 50 에폭 파인튜닝 [요약]
- 대조: 소거 실험에서 "Pruned"(보상 없음) / "LIF"(원 LIF 로 파인튜닝) / "sLIF"(제안) 비교 [요약]. 이는 *파인튜닝을 한 대조(LIF)* 가 있다는 뜻이나,
  가지치기 안 한 원 모델을 같은 에폭 파인튜닝한 대조는 아니다. 요약 도구가 "does not explicitly describe" 라고 답함. 원문 대조 미완.
- 우리와의 대응: 약함. 미게재 프리프린트라 방패로는 부적합.

### A3. Activity Pruning for Efficient Spiking Neural Networks
- 저자: Tong Bu, Xinyu Shi, Zhaofei Yu [NeurIPS 페이지 확인]
- 학회·연도: NeurIPS 2025
- URL: https://neurips.cc/virtual/2025/poster/115023 / 코드 https://github.com/putshua/Activity-Pruning-SNN
- 사전학습 체크포인트 파인튜닝 여부, 대조 설계: **미확인** (초록과 README 에 언급 없음. README 는 `main.py --config=cifar10.yaml` 학습 스크립트만 안내). 원문 PDF 를 열어 보지 못함. 선례로 인용 금지, 원문 확인 후 판단.

### A 에서 찾아봤으나 해당 안 된 것
- Quantized Spike-driven Transformer (arXiv 2501.13492): 직접 학습 QAT, 사전학습 FP 체크포인트 출발 근거 없음 [요약]. 제외.
- SpikeZIP-TF (arXiv 2406.03470): ANN ReLU 파인튜닝 후 QAT. 같은 조건 대조 여부 미확인. 제외.

---

## B. 일반 딥러닝 (확인된 것 8편)

### B1. ProSparse (가장 구조가 비슷함: 사전학습 모델 + 규제 항 + 같은 예산 무규제 대조)
- 저자·소속: Chenyang Song, Xu Han, Zhengyan Zhang 외, Tsinghua / CAS ICT / Tencent [원문]
- 학회·연도: COLING 2025 (검색 결과가 "31st International Conference on Computational Linguistics" 로 표기) [요약]. arXiv 2402.13516v7
- URL: https://arxiv.org/abs/2402.13516
- 대조 (원문, 부록 H): "Given the different amounts of training tokens, we compare ProSparse with the other two baselines (i.e., vanilla ReLU and shifted ReLU) in terms of the average sparsity and performance." 두 baseline 은 "regularization-free" 이고 같은 누적 토큰 수에서 비교 (표 4: 34.60B 와 89.13B) [원문].
- 수치 (표 4, LLaMA2-7B, 7B 설정): 34.60B 토큰에서 Vanilla ReLU 희소성 66.04% / 성능 41.40, Shifted ReLU 69.59% / 41.33, **ProSparse 89.32% / 38.46**. 89.13B 에서 64.93%/41.52, 68.35%/41.40, **88.29%/40.67** [원문].
- 우리와의 대응: 출발 체크포인트 동일, L1 규제 (우리는 sc_rate 가중 L2) 의 유무가 유일한 차이인 대조. 우리 (a)/(b) 와 같은 논리 구조.
- **불리한 점**: 규제 없는 대조의 정확도가 더 높았다. 우리가 "정확도 희생 없음" 을 주장하려면 ProSparse 처럼 희소성을 같은 축에 두고 비교해야 하며 정확도만 보면 대조가 이길 수 있다. 또 대조는 규제 없는 파인튜닝이 아니라 ReLU 치환 포함이라 순수한 "방법 유무" 대조는 아니다.

### B2. Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
- 저자·소속: Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, Danqi Chen, Princeton [저자 목록은 기억 기반, 원문 첫 페이지 대조 안 함: 미확인]
- 학회·연도: ICLR 2024
- URL: https://arxiv.org/abs/2310.06694
- 대조 (원문, 부록, Figure 9): "we compared Sheared-Pythia-160M against keeping pre-training the Pythia-160M model with the same amount of tokens. ... continuing pre-training Pythia-160M starts off performing better, however, the Sheared-Pythia-160M learns faster and eventually exceeds the performance of continuing pretraining on Pythia-160M. These are some very preliminary results" [원문].
- 추가 (부록 F.7, Figure 11/12): 같은 크기의 기존 모델 INCITE-Base-3B 를 계속 사전학습한 것과 Sheared-LLaMA-2.7B 비교 [원문].
- 우리와의 대응: "원 체크포인트를 같은 토큰으로 계속 학습한 대조" 의 정석. 단 Pythia-160M 한 모델, 저자 본인이 preliminary 라 표기.
- 불리한 점: 이 대조는 주 결과가 아니라 부록. 비교 대상이 *구조가 같은 계속 학습* 이 아니라 pruning 으로 작아진 모델과의 비교 (크기가 다름).

### B3. INTERLACE: Interleaved Layer Pruning and Efficient Adaptation in Large Vision-Language Models
- 저자·소속: Parsa Madinei 외, UC Santa Barbara [원문]
- 학회·연도: arXiv 2511.19676 (2025-11). 학회 **미확인**
- URL: https://arxiv.org/abs/2511.19676
- 대조 (원문 4.3): "we fine-tune the last 25% of layers in the 8B model on the same 1% FineVision subset for the same single epoch, matching the exact training configuration used for our pruned models. This ablation allows us to determine whether performance improvements stem from our strategic layer selection or simply from additional training on high-quality data." [원문]
- 수치 (부록 A.4 표 6, Qwen3-VL-8B, CoT 16384 토큰): 가지치기 안 한 dense 평균 84.3% → 같은 파인튜닝 후 81.6% (-2.7%p) [원문].
- 우리와의 대응: 리뷰어 질문 "더 학습해서 좋아진 것 아니냐" 와 문장까지 같다. 가장 직접적인 문구 선례. 우리 질문과 정확히 같은 목적.
- 불리한 점: 프리프린트, VLM, 층 일부만 학습. 위상이 낮다.

### B4. Sparse Fine-tuning for Inference Acceleration of Large Language Models (SquareHead)
- 저자: Eldar Kurtic, Denis Kuznedelev, Elias Frantar, Michael Goin, Dan Alistarh [요약+원문 일부]
- 연도·학회: arXiv 2310.06927 (2023-10). NeurIPS 2023 페이지 (neurips.cc/virtual/2023/81168) 가 검색되었으나 본회의/워크숍 여부 **미확인**
- URL: https://arxiv.org/abs/2310.06927
- 대조 (원문): "we fine-tune MPT-7B via SFT to obtain a highly accurate and competitive dense baseline" 이후 이것을 교사 겸 기준선으로 놓고 희소 파인튜닝. Figure 4 의 "Dense fine-tuned" 선이 같은 데이터 (GSM8K) 파인튜닝 대조 [원문].
- 수치 (표 3, MPT-7B, GSM8K, FP32): 희소 0% 28.2, 40% 32.9, 50% 30.6, 70% 28.0, 80% 23.1 [원문]. 원문 해석: "distilled models can outperform the dense baseline at low sparsity" [원문].
- 우리와의 대응: 사전학습 모델을 *같은 데이터로 파인튜닝한 dense* 를 기준으로 삼음. 우리 (a) 와 같은 역할.
- 불리한 점: 과제 파인튜닝 (GSM8K) 이지 사전학습 분포 계속 학습이 아님.

### B5. Compact Language Models via Pruning and Knowledge Distillation (Minitron)
- 저자·소속: Saurav Muralidharan, Sharath Turuvekere Sreenivas 외 (NVIDIA) [원문]
- 학회·연도: NeurIPS 2024 (proceedings.neurips.cc 에 같은 제목 PDF 존재) [검색 결과]. arXiv 2407.14679
- URL: https://arxiv.org/abs/2407.14679
- 대조 (원문 표 11): 같은 계산량(iso-compute) 에서 4B-Random-Init (150B 토큰) / 4B-Pruned (150B) / 4B-Pruned-Distill (100B). "we compare approaches under iso-compute settings." [원문]
- 수치 (표 11, 4B): 랜덤 초기화 150B MMLU 24.36, 가지치기 150B 24.57, 가지치기+증류 100B 37.81 [원문].
- 우리와의 대응: **같은 예산 비교의 관행** 근거. 단 대조는 "같은 체크포인트 계속 학습" 이 아니라 랜덤 초기화.
- 불리한 점: 직접 대응 약함. 보조 인용용.

### B6. ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models
- 저자·소속: Iman Mirzadeh 외, Apple [검색 결과 확인]
- 학회·연도: ICLR 2024
- URL: https://arxiv.org/abs/2310.04564
- 방식 (원문): 사전학습 Falcon 7B / Llama 7B / OPT 6.7B 에 ReLU 치환 후 파인튜닝, "For finetuning the pretrained models, we follow the original pretraining recipe, except we use a fixed learning rate of 1.5e-5"; 30B / 50B 토큰 [원문].
- 대조: 표 2 (MMLU 5-shot) 에서 SiLU*/GELU/ReLU 각각 파인튜닝한 모델을 나란히 비교 ("Models finetuned with different activation functions have similar performance"). 단 Llama 7B 의 SiLU* 줄이 원 체크포인트인지 같은 예산으로 파인튜닝한 것인지는 원문만으로 확정 못 함: **미확인**.
- 우리와의 대응: 사전학습 체크포인트 + 짧은 저학습률 파인튜닝으로 희소성 이득을 주장한 ICLR 선례 (프로토콜 근거).

### B7. Accelerating Sparse Deep Neural Networks (2:4, NVIDIA ASP)
- 저자: Asit Mishra 외, NVIDIA [원문]
- 연도: arXiv 2104.08378 (2021)
- URL: https://arxiv.org/abs/2104.08378
- 방식 (원문): "retrain the pruned model: ... use the same optimizer and schedule (learning-rate, schedule, number of epochs, etc.) as in Step 1" [원문].
- 우리와의 대응: 재학습 예산을 원 학습과 동일하게 고정해 *"방법 외 변수는 같다"* 를 확립. 단 dense 를 추가 학습한 대조는 아님. 약한 선례.

### B8. Comparing Rewinding and Fine-tuning in Neural Network Pruning (Renda, Frankle, Carbin)
- ICLR 2020, https://arxiv.org/abs/2003.02389 [검색 결과 확인]
- 파인튜닝 스케줄 (학습률 선택) 이 결과를 좌우함을 보임. 우리 대조가 같은 학습률·스케줄을 써야 한다는 근거. 본문 인용구는 확인 못 함 (요약만).

### B 보조
- What is the State of Neural Network Pruning? (Blalock 외, MLSys 2020) https://arxiv.org/abs/2003.03033: 81편 메타분석. 표준 벤치마크 부재를 지적 [요약]. 비교 프로토콜 표준화 필요성의 근거. 직접 인용구 미확인.
- Wen 외, Structured Sparsity Learning (NeurIPS 2016): baseline 에서 시작해 규제 학습 후 파인튜닝 [요약]. 대조 설계는 우리와 다르므로 제외.

### B 에서 대조가 **없는** 것으로 확인된 것 (인용하면 역공격 소지)
- Enabling High-Sparsity Foundational Llama Models (arXiv 2405.03594): dense 를 같은 토큰으로 계속 사전학습한 비교가 없음 [요약, 원문 검색 확인 못 함].
- Small LLMs: Pruning vs. Training from Scratch (arXiv 2606.14150): dense 체크포인트 계속 학습 대조 없음, 스크래치와 비교 [요약].
- Continual LLM Upcycling (arXiv 2606.10722): 명시적 대조 확인 못 함.

---

## C. 가장 강한 선례 3편

1. **ProSparse (COLING 2025)** — 사전학습 체크포인트 + 규제 항 + 같은 누적 토큰의 규제 없는 대조. 구조가 우리와 같다. 단 대조가 더 높은 정확도를 낸 사실을 감안할 것.
2. **INTERLACE (arXiv 2025)** — "더 학습해서 좋아진 것 아니냐" 를 막기 위해 같은 데이터·같은 에폭 파인튜닝 대조를 둔다고 문장으로 밝힘. 문구상 가장 직접적이나 프리프린트.
3. **Sheared LLaMA (ICLR 2024)** — 같은 토큰으로 원 체크포인트를 계속 사전학습한 대조 (Figure 9, 부록). 탑 학회지만 부록·preliminary.

SNN 쪽 방어선: TP-Spikformer (ICLR 2026) 가 공식 SDT-V3 체크포인트를 lr 1e-5, 50 에폭으로 파인튜닝하는 *프로토콜* 의 선례 (대조는 없음).

## D. 리뷰어 답변 초안

영문:
> Following common practice in pruning and sparsification of pretrained models (e.g., Sheared-LLaMA, ICLR 2024; ProSparse, COLING 2025), we compare against a control that is fine-tuned from the same official SDT-V3 checkpoint for the same number of epochs with identical optimizer and schedule but without the regularizer, so the regularizer is the only difference. The fine-tuning protocol itself (official checkpoint, low learning rate, tens of epochs) follows prior SDT-V3 compression work (TP-Spikformer, ICLR 2026), which reports against the official checkpoint only; our same-budget control is therefore strictly more conservative, and any gain over it cannot be attributed to additional training.

국문:
> 사전학습 모델을 가지치기·희소화하는 기존 연구 (Sheared-LLaMA, ICLR 2024; ProSparse, COLING 2025) 의 관행에 따라, 같은 공식 SDT-V3 체크포인트에서 같은 에폭 수, 같은 옵티마이저·스케줄로 파인튜닝하되 규제 항만 뺀 대조 런과 비교했다. 규제 항이 유일한 차이다. 파인튜닝 프로토콜 (공식 체크포인트, 낮은 학습률, 수십 에폭) 자체는 SDT-V3 압축 선행연구 (TP-Spikformer, ICLR 2026) 를 따랐다. 이 논문은 공식 체크포인트와만 비교하므로, 우리의 같은 예산 대조는 더 보수적이며 그 위의 이득은 추가 학습으로 설명되지 않는다.

주의: 위 초안은 "SNN 선례가 대조를 썼다" 고 말하지 않는다. 그렇게 쓰면 거짓이 된다. 초안에서 Sheared-LLaMA 를 쓸 때는 "소규모 부록 실험" 임을 감안하고, 본문에 한 줄 더 쓸 여유가 있으면 INTERLACE (arXiv) 를 두 번째 근거로 병기.

## E. 후속 확인이 필요한 것
- Activity Pruning (NeurIPS 2025) 원문에서 사전학습 출발 여부와 대조 확인 (A3)
- Sheared LLaMA 저자 목록 대조, Blalock/Renda 에서 직접 인용 가능한 문구 확보
- ReLU Strikes Back 표 2 의 SiLU* 줄이 파인튜닝한 대조인지 확인 (B6)
- ProSparse 의 게재 학회 (COLING 2025) 를 ACL Anthology 에서 재확인
