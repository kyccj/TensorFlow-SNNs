# 활동량 가중 스파이크 규제의 예비 분석: 성능 비교와 역전파 구현

*A Preliminary Study of Activity-Weighted Spike Regularization: Performance and Backward Implementation*

> 검토용 국문 초안 · 2026-09-16 · 본문 4쪽 short paper를 목표로 한 구성.
> 투고처는 AAAI 2027의 SNN 관련 워크숍을 후보로 둔다. 개최·분량·출판 방침은 해당 CFP에서 확정해야 한다.
> 아래 결과는 `docs/obsidian/EIP-results-26-09-16.md`의 잠정 집계를 인용한다. 기존 원고의 동결 집계와 다르며, 최종 제출용 결과로 확정한 것이 아니다. `[확인 필요]`는 편집 메모다.
> 증거 출처, 개별 런, 제출 전 해결 사항은 [작성 메모](workshop-evidence-notes.md)에 분리했다. 기존 CVPR 원고와 Method 절은 수정하지 않는다.

## 초록

스파이킹 신경망의 발화량을 줄이는 규제는 정확도와 계산 비용의 절충을 조절한다. 그러나 활동량에 따라 뉴런별 규제 강도를 다르게 주는 설계가 단순한 규제보다 유리한지는 별도의 비교가 필요하다. 본 연구는 ResNet-19에서 무규제 baseline, plain L2, 활동량 가중 규제인 ours를 비교하는 예비 실험을 보고한다. CIFAR-10의 현재 집계에서 ours는 baseline 대비 기록된 스파이크 지표를 51.4% 줄였으며, 평균 검증 정확도는 96.71%(n=3)에서 96.67%(n=7)로 변했다. 이 감소율은 집계 경로 간 불일치가 남아 있는 잠정 측정치다. 스파이크 지표를 보정한 탐색적 회귀에서 plain L2 대비 정확도 오프셋의 95% 신뢰구간은 0을 포함했다(n=27). 구현을 점검한 결과, 사용자 정의 역전파와 외부 가중치 곱셈의 합성으로 스파이크 변수에 전달되는 규제 신호가 의도했던 선형 가중치가 아닌 제곱 가중치를 포함함을 확인했다. 우리는 기존 결과를 이 실제 구현의 결과로 기술하고, 순전파 규제값과 역전파 규칙을 구분한다. 현재 결과는 성능 우위나 에너지 절감의 실측 증거가 아니며, 가중치 지수의 효과를 분리하기 위한 비교 실험의 출발점이다.

## 1. 서론

SNN의 효율성을 평가하려면 분류 정확도와 함께 발화 활동을 측정해야 한다. 스파이크 수가 줄면 이벤트 기반 연산을 줄일 가능성이 있지만, 그 감소가 장치의 전력이나 실행 시간에 얼마나 반영되는지는 연결 구조와 실행 방식에 달려 있다. 선행 연구는 시냅스 연산량 규제, 스파이크 희소성 규제, 뉴런·가중치 가지치기, 역치 적응 등 여러 접근을 제시했다 [1–4]. 따라서 발화량이 감소했다는 사실만으로 새로운 규제 설계의 필요성이 입증되지는 않는다.

본 연구의 질문은 활동량으로 가중한 규제가 단순한 plain L2와 비교해 어떤 결과를 만드는가이다. 무규제 baseline은 규제를 추가한 전체 효과를 보여주지만, 활동량 가중 설계 자체의 가치를 평가하려면 단순 규제와의 비교가 필요하다. 우리는 정확도와 기록된 스파이크 지표를 함께 비교하고, 차이가 확인되지 않은 부분도 보고한다.

이 질문을 다루는 과정에서 구현과 설명 사이의 차이를 발견했다. 순전파에서 사용하는 가중치가 사용자 정의 역전파 안에서도 사용되면서, 스파이크 변수에 전달되는 규제 신호에는 가중치가 두 번 곱해졌다. 기존 학습 결과를 이해하려면 이 합성 경로를 명시해야 한다. 또한 이 차이가 학습 후 활동 분포를 설명하는지는 별도의 대조 실험으로 검증해야 한다.

이 예비 연구는 두 가지를 보고한다. 첫째, 현재 실험 범위에서 발화 감소와 plain L2 대비 성능 우위를 구분한다. 둘째, 실행된 역전파 규칙을 기술하고 선형·제곱 가중치를 분리해 검증할 수 있는 비교를 정의한다. 구현상의 발견 자체를 새로운 학습 알고리즘의 우수성으로 주장하지 않는다.

## 2. 관련 연구

Sorbaro 등은 시냅스 연산량을 고려한 학습으로 SNN의 에너지 비용을 줄이는 접근을 제시했다 [1]. Yan 등의 BPSR는 스파이크 희소성 규제를 포함한다 [2]. 본 연구의 plain L2는 로컬 구현의 비교군 이름이며, BPSR 전체 방법의 재현이나 문헌 방법 일반을 대표하는 이름으로 사용하지 않는다.

Shi 등은 비구조적 가중치·뉴런 가지치기를 결합하는 프레임워크를 제안했다 [3]. Bu 등의 Activity Pruning은 AT-LIF 뉴런과 역치 적응을 이용해 발화 활동을 억제한다 [4]. 이러한 연구와의 관계에서 우리는 활동 감소 자체의 최초성이나 가지치기 성능 우위를 주장하지 않는다. 본문은 동일 코드베이스 내 비교와 역전파 구현 분석에 범위를 둔다. 문헌 방법과의 동일 조건 성능 비교는 아직 완료되지 않았다.

## 3. 비교 대상과 실제 역전파

### 3.1 활동량 가중치

뉴런 i의 시간 단계 t에서의 이진 스파이크를 sᵢ,ₜ, 전체 T단계의 누적 스파이크를 qᵢ = Σₜsᵢ,ₜ라 한다. 현재 ours 설정은 마지막 시간 단계에 누적 스파이크를 대상으로 규제를 적용한다. 가중치 계산에 쓰는 활동값 aᵢ는 발화한 뉴런에서는 누적 스파이크 수이고, 발화하지 않은 뉴런에서는 임계값으로 정규화하고 [0,1]로 자른 막전위 값이다. 후자의 항은 규제용 가중치 계산에서 미분하지 않는다.

합성 규칙을 설명하기 위해 기본 설정의 가중치를 다음과 같이 쓴다.

\[
w_i=1-\operatorname{divide\_no\_nan}\left(a_i,\max_{j\in G(i)}a_j\right).
\]

G(i)는 같은 입력·채널의 공간 위치 집합이다. 최대값이 0이면 나눗셈 결과를 0으로 정의한다. 이 설명은 기본 가중치 계수 1, 추가 제곱·평균 정규화·분모 바닥값을 사용하지 않는 경우다. 각 기존 런이 이 설정을 사용했는지는 저장된 설정으로 대조해야 한다. `[확인 필요: 런별 설정 및 비합성곱 층의 정규화 축]`

가중치 계산용 활동 상태와 미분 가능한 누적 스파이크 q는 코드에서 구분된다. 이하의 식에서는 w를 고정값으로 취급한다.

### 3.2 순전파와 사용자 정의 역전파

한 규제 블록에서 x = w ⊙ q라 두면, 순전파에서 보고되는 값은

\[
R(q,w)=\|w\odot q\|_2
\]

이다. 코드의 노름은 해당 텐서 전체를 합산하므로 배치와 층의 크기도 값에 영향을 준다. 전체 학습 손실은 과제 손실과 각 블록의 규제값을 계수 λ로 가중한 합으로 구성된다.

현재 기본 구현은 외부에서 x를 만든 뒤, 사용자 정의 함수가 x에 대해 wᵢ/R이라는 역전파 규칙을 돌려준다. 외부 곱셈의 미분을 합성하면, R>0에서 qᵢ가 받는 규제 신호는

\[
g_i^{(2)}=\lambda\frac{w_i^2}{R}.
\]

여기서 g는 **사용자 정의 역전파 신호**다. 순전파 식 R의 통상적인 도함수라고 표기하지 않는다. w를 고정했을 때 λR의 통상적인 도함수는 λwᵢ²qᵢ/R이며, 현재 규칙은 qᵢ 인자를 생략한다. 따라서 블록의 R이 양수이고 wᵢ>0이면, qᵢ=0인 뉴런에도 이 경계에서 신호가 전달될 수 있다. 최종 매개변수 기울기는 여기에 스파이크 생성 함수의 대리 미분과 앞선 계산 경로를 합성한 값이다. 블록 전체 R=0인 경우 구현은 반환 신호를 0으로 선택한다.

선형 가중치 비교 경로는 동일한 q,w에 대해 같은 순전파 R을 반환하면서, q를 직접 입력받아

\[
g_i^{(1)}=\lambda\frac{w_i}{R}
\]

을 전달한다. 현재 집계에 포함된 ours 결과는 기존 제곱 가중치 경로의 결과이며, 선형 경로의 학습 결과는 이 초안에 포함하지 않는다.

같은 q,w,λ에서 0<wᵢ<1이면 gᵢ⁽²⁾/gᵢ⁽¹⁾=wᵢ다. 이는 그 지점에서의 규제 신호 차이를 설명한다. 실제 학습에서는 q,w,R,λ가 함께 변하므로, 이 식만으로 학습 후 정확도나 활동 집중도의 차이를 예측하거나 원인을 확정할 수는 없다.

### 3.3 plain L2와 비교할 때의 범위

plain L2는 각 시간 단계의 스파이크 텐서에 L2 노름과 통상적인 노름 역전파를 적용하고 고정 계수를 사용한다. ours는 활동량 가중치뿐 아니라 누적 스파이크에 대한 마지막 단계 적용, 사용자 정의 역전파, 손실 비율에 따른 계수 갱신도 포함한다. 따라서 두 방법의 차이는 **전체 학습 절차의 비교**이며, 가중치 하나의 효과를 분리한 실험이 아니다.

ours의 계수는 직전 에폭의 과제 손실과 규제값을 이용해 목표 비율 ρ에 맞추어 갱신한다. ρ 자체는 선택해야 하는 하이퍼파라미터다. 순전파 손실 비율이 같아도 역전파 신호의 크기나 방향이 같아지는 것은 아니므로, 이를 튜닝 불필요 또는 기울기 균형의 보장으로 해석하지 않는다. `[확인 필요: 런별 비율 모드·초기값·상한과 손실 집계 정의]`

## 4. 예비 실험

### 4.1 설정과 보고 범위

주 실험은 ResNet-19와 CIFAR-10이며 시간 단계는 T=4다. 무규제 baseline, plain L2, ours를 비교한다. 정확도는 현재 로그의 val_acc이고, 스파이크 지표도 각 런에서 val_acc가 최대인 에폭의 값을 읽었다. 이를 독립된 최종 테스트 평가라고 부르지 않는다. 반복 번호는 고정 시드가 아니므로 표준편차는 런 간 변동이다.

현재 집계는 진행 중인 실험의 스냅샷이다. 기존 원고의 상위 4개 선별·동결 집계와 런 집합이 다르며, 이 차이를 해결하기 전에는 최종 성능 표로 확정할 수 없다. 특히 미완주 런이 plain L2와 BPSR 조건에 몰려 있어 완주한 런만으로 얻은 비교가 전체 실험 계획을 대표하지 않을 수 있다.

### 4.2 발화 감소와 정확도

ours의 ρ=3×10⁻³ 설정은 평균 검증 정확도 96.67%, 런 간 SD 0.128%p(n=7)를 기록했다. baseline은 96.71%, SD 0.108%p(n=3)다. 보고된 평균 차이는 −0.04%p이며, 현재 로깅 지표에서의 스파이크 감소율은 51.4%다. 이는 유사한 정확도 범위에서 발화량을 낮춘 관찰이지만, 비열등성이나 무손실을 통계적으로 입증한 결과는 아니다.

스파이크 계수기와 뉴런별 기록의 합계 사이에 약 11% 불일치가 남아 있다. 위 상대 감소율은 동일 로깅 경로 내 잠정 비교이며, 절대 스파이크 수와 에너지 환산값은 제시하지 않는다. 집계 불일치가 방법별로 동일하게 발생하는지도 확인되지 않았으므로, 감소율은 통일된 측정으로 재확인해야 한다.

plain L2의 λ=2.5×10⁻⁷과 5×10⁻⁷ 설정은 각각 96.59%, SD 0.125%p(n=5)와 96.49%, SD 0.098%p(n=5)를 기록했다. ours의 ρ=4×10⁻³ 설정은 96.59%, SD 0.105%p(n=5)였다. 이 값들은 서로 다른 발화 수준의 동작점이므로 정확도만 나열해 방법 간 순위를 매기지 않는다. 개별 런과 훈련·검증 손실은 내부 증거표에 보존했다.

`[그림 1 예정: 통일된 계수기로 다시 확인한 실제 스파이크 수(x)와 검증 정확도(y). 조건별 모든 런을 표시한다. 현재는 절대 수치 보류로 그림을 만들지 않는다.]`

### 4.3 plain L2 대비 우위는 확인되지 않았다

현재 집계는 다음 탐색적 회귀를 보고한다.

\[
\mathrm{acc}=a+b\log S+c\,\mathbb{1}[\mathrm{ours}].
\]

여기서 S는 동일 로깅 경로의 스파이크 지표다. CIFAR-10에서 c의 추정값은 +0.071%p이며 95% 신뢰구간은 [−0.021, +0.163]%p, p=0.123이다(n=27). 이 분석은 ours의 정확도 우위를 확인하지 못했다. 반대로 두 방법의 동등성을 입증한 것도 아니다. 로그-선형 형태, 양 방법에 공통인 기울기, 미완성 스윕, 체크포인트 선택 및 스파이크 계수에 의존하는 잠정 분석이다.

CIFAR-100의 탐색적 비교에서는 c=−0.150%p, 95% 신뢰구간 [−0.699, +0.399]%p였다(n=7). 표본이 작고 스윕 범위도 제한되어 있지만, CIFAR-10의 양의 점추정이 다른 데이터셋에서도 재현됐다고 말할 근거는 없다.

### 4.4 아직 답하지 못한 질문

현재 자료만으로는 제곱 가중치가 선형 가중치보다 유리한지, 활동 분포 차이가 가중치 지수 때문에 생겼는지 판단할 수 없다. 이를 위해서는 나머지 설정을 유지한 두 역전파 경로의 비교가 필요하다. 같은 ρ에서의 비교는 동일 제어 설정의 효과를, 각각의 강도를 스윕한 비교는 도달 가능한 정확도–발화 절충을 다룬다. 두 질문을 구분해야 한다.

뉴런별 활동 분석은 유망한 후속 질문이지만, 기존 기록에는 평가 모드·입력 집합·반복 수를 재확인해야 하는 값이 섞여 있다. 따라서 이번 초안은 과거의 활성 뉴런 감소율이나 단일 체크포인트의 집중도를 확정 기여로 옮기지 않는다. 특히 활동량이 높은 뉴런을 보존했다는 관찰과 과제에 중요한 뉴런을 보존했다는 주장은 다르다. 후자는 뉴런 제거에 따른 성능 변화 등의 별도 검증이 필요하다.

## 5. 논의와 한계

현재 결과에서 가장 분명한 구분은 무규제 모델 대비 발화 감소와 단순 규제 대비 설계 이득 사이에 있다. 전자는 관찰되었으나 측정 재확인이 필요하고, 후자는 현재 비교로 입증되지 않았다. 이 결과는 활동량 가중 설계가 불필요하다는 결론도, plain L2보다 우수하다는 결론도 지지하지 않는다.

역전파의 제곱 가중치를 발견했다고 기존 실험이 자동으로 무효가 되는 것은 아니다. 다만 그 실험은 실제 실행된 규칙에 대한 결과이며, 의도했던 선형 규칙의 증거로 재사용할 수 없다. 제곱 가중치를 사후적으로 더 좋은 설계였다고 설명하는 것 역시 직접 비교 없이는 정당화되지 않는다.

본 연구는 한 아키텍처와 제한된 데이터셋에서 수행한 시뮬레이션이다. 문헌 방법의 직접 재현 비교, 선형·제곱 가중치의 학습 비교, 손실 비율 제어기의 독립적 효과, 하드웨어 전력·지연시간 검증은 완료되지 않았다. 활동 집중은 구조적 가지치기 가능성이나 배포 이득과 같지 않다. 이 범위에서 본 연구는 추가 검증이 필요한 경험적 관찰과 구현 분석을 제공한다.

## 6. 결론

활동량 가중 스파이크 규제를 평가하려면 무규제 baseline 대비 효과, 단순 규제 대비 이득, 실제 역전파 규칙을 함께 확인해야 한다. 현재 실험은 발화 지표 감소를 보였지만 plain L2 대비 정확도 우위를 확인하지 못했다. 구현 점검은 기존 결과가 제곱 가중치 역전파로 얻어졌음을 드러냈다. 선형 가중치와의 대조 실험 및 통일된 활동 측정이 이 설계의 가치를 판단하는 다음 단계다.

## 참고문헌

1. Sorbaro et al. (2020). *Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications*. Frontiers in Neuroscience. [원문](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full)
2. Yan et al. (2022). *Backpropagation With Sparsity Regularization for Spiking Neural Network Learning*. Frontiers in Neuroscience. [원문](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.760298/full)
3. Shi et al. (2024). *Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework*. ICLR. [원문](https://proceedings.iclr.cc/paper_files/paper/2024/file/dd3c889922df2112a5b1769e3c19e28e-Paper-Conference.pdf)
4. Bu, Shi, and Yu (2025). *Activity Pruning for Efficient Spiking Neural Networks*. NeurIPS. [공식 초록](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fa12d67b5939c37ea8a4659c31a08d2c-Abstract-Conference.html)

## 영문 초록 — 검토용

Spike regularization offers a way to trade classification accuracy for reduced activity in spiking neural networks. Whether activity-dependent weighting improves this trade-off over a simple penalty remains an empirical question. We present a preliminary comparison of an unregularized baseline, a plain L2 penalty, and an activity-weighted training procedure on ResNet-19. In the current CIFAR-10 snapshot, the weighted procedure reduces the logged spike metric by 51.4%, while mean validation accuracy changes from 96.71% (three runs) to 96.67% (seven runs). The activity measurement remains provisional because two counting paths disagree. An exploratory regression over 27 runs does not establish an accuracy advantage over plain L2 after accounting for the logged spike metric. An implementation audit further identifies squared, rather than intended linear, activity weights in the backward signal due to the composition of external multiplication and a custom backward rule. We distinguish the executed rule from the forward regularization value and describe the comparison needed to isolate the weighting exponent. These observations do not establish superiority, statistical equivalence, or measured hardware energy savings.
