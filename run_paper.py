"""논문용 통합 런처 (26-09-03).

기존 런처들은 원본 config 의 특정 줄을 정규식으로 갈아끼웠다. 방법이 하나뿐일 때는 됐지만
조합 x 방법 x 손잡이 x 시드로 늘어나면 치환 대상이 매번 달라져 못 버틴다.
(`run_gap3.py` 는 `verify()` 가 `a`/`b` 미정의인 채로 남아 job 을 조용히 죽였다.)

**대신 `config.set()` 바로 앞에 덮어쓰기 블록을 끼워 넣는다.** 파이썬은 나중 대입이 이기므로
앞쪽 설정이 무엇이든 이 블록이 최종값이 된다. 원본은 GPU 줄과 exp_set_name 만 건드린다.

    python run_paper.py r19c10 prop 1e-3 --seeds 1,2,3,4,5
    python run_paper.py r19c10 sm 1e-7 --seeds 1,2,3 --gpus 0,2
    python run_paper.py --list

**방법**
    prop        제안법 — 채널 내 1-maxnorm + vmem(silent_only, gain=1.0) + final_step + loss-ratio(rho)
    ours_w1     prop 와 전부 같고 wta_rev 의 spike 기울기만 w 한 번 (기본은 w^2). knob 은 rho
    ours_ch     prop 와 전부 같고 max 범위만 채널 합끼리 ('channel'). knob 은 rho
    ours_ch_b7  ours_ch + shape_beta=0.7 (1등 채널도 w=0.3 을 받는다). knob 은 rho
    ours_ch_b4  ours_ch + shape_beta=0.4. knob 은 rho
    ours_ch_taylor ours_ch 와 전부 같고 채널 서열만 Taylor 중요도 |sum (dL/ds)*s| 의 EMA 로 매긴다
                (규제 항은 여전히 스파이크. 26-10-02). knob 은 rho
    ours_chxwc  ours_ch 와 전부 같고 가중만 inter x intra 곱: (1-ch/ch_max)*(1-scp/max_hw scp)
                (maxnorm_group='channel_x_within'. 26-10-05). knob 은 rho
    sm          1-softmax + 고정 lambda        (내부 최강 기준선)
    l2          plain L2 + 고정 lambda         (문헌 기준선. reg_spike_out_sc=False 로 두면
                                                neurons.py:1238 의 `else: # old - previous work`
                                                가지로 가서 l2_norm(spike) 이 된다)
    l2_lr       plain L2 + loss-ratio          (l2 와 전부 같고 lambda 만 제어기가 푼다. knob 은 rho.
                                                제어기의 몫과 제안법 나머지의 몫을 가르는 대조군)
    suetake     Suetake et al. (arXiv 2302.01500) synaptic interaction penalty + 고정 lambda
                (lambda * sum_l psi_l * sum_i S_i / B, psi_l = fan-out, lambda 는 에폭 비례 선형
                 램프. 경쟁·maxnorm·vmem·wta_rev·loss-ratio 없음. 26-10-04). knob 은 lambda
    unisparse   UniSparse 스파이크 단독 (Shi, Bu, Yu ICML 2026, --lmbda-weight 0). 벌점
                g_s * sum_c mean s 의 기울기를 각 층 바로 앞 BN 의 gamma·beta 로만 보내 옵티마이저 밖에서
                lambda*lr 로 갱신 (원 코드 방식). 가중치는 과제 손실만 받는다 (lib_snn/thr_reg.py, 26-10-20).
                knob 은 lambda
    unisparse_opt  unisparse 와 같고 벌점 몫 lambda*dE 를 gamma·beta 기울기에 더해 AdamW 로 (식 14 목적 충실,
                reg_spike_unisparse_update='optimizer'). 착지 시험 비교용. knob 은 lambda
    atlif       Activity Pruning AT-LIF (Bu, Shi, Yu NeurIPS 2025). 은닉층 출력 theta*s (층당 학습 theta),
                theta 는 AdamW 밖에서 식 12 (theta -= lr*dL/dtheta, += lambda*lr*규제 통계). 가중치는 과제
                손실만 (lib_snn/thr_reg.py, 26-10-20). knob 은 lambda_theta
    atlif_regonly  atlif 와 같고 theta 의 과제 몫을 버린다 (규제로만 움직임, 원 코드와 사실상 같음.
                reg_spike_atlif_theta_task='none'). 착지 시험 비교용. knob 은 lambda_theta
    abl_nofinal 제안법에서 final_step 만 끔     (시점의 기여)
    abl_novmem  제안법에서 vmem 만 끔 (gain=0)  (막전위의 기여)
    abl_noinv   제안법에서 1- 반전만 끔         (maxnorm_plain = sc/max. 이름의 근거)
    abl_noinv_wc 제안법에서 1- 반전**만** 끈다. 채널 내 묶음과 vmem 은 그대로 둔다.
                abl_noinv 는 maxnorm_plain 가지라 group·vmem 을 안 읽어 세 요소가 동시에
                빠진다 -- 어블레이션 표의 '반전의 효과'는 이 팔이다. knob 은 rho
    abl_nolr    제안법에서 loss-ratio 만 끔  (고정 lambda. knob 이 rho 가 아니라 lambda)
                (위 abl_* 다섯 팔은 옛 제안법 prop(=ours_intra_ch)에서 뺀 것이다)

    [ours_layer_w1 어블레이션 -- 09-23 확정 제안법 ours_layer_w1 에서 한 요소씩 뺀다]
    lw1_novmem  ours_layer_w1 에서 vmem 만 끔 (gain=0)   knob 은 rho
    lw1_nofinal ours_layer_w1 에서 final_step 만 끔       knob 은 rho
    lw1_nolr    ours_layer_w1 에서 loss-ratio 만 끔       knob 은 **lambda** (고정)
    lw1_noinv   ours_layer_w1 에서 1- 반전만 끔 (no_inv)  knob 은 rho

    [ours_ch 어블레이션 -- ours_inter_ch(run 키 ours_ch) 에서 한 요소씩 뺀다 (26-10-04)]
    ch_novmem   ours_ch 에서 vmem 만 끔 (gain=0)          knob 은 rho
    ch_nofinal  ours_ch 에서 final_step 만 끔            knob 은 rho
    ch_noinv    ours_ch 에서 1- 반전만 끔 (no_inv)       knob 은 rho
    ch_nolr     ours_ch 에서 loss-ratio 만 끔            knob 은 **lambda** (고정)
    ch_nowtarev ours_ch 에서 wta_rev backward 만 끔 (기울기 = ||S*w|| 의 자동미분)  knob 은 rho (26-10-28)

    [규제 시작 지연 -- 원 방법과 전부 같고 loss-ratio 의 start_ep 만 다르다. knob 은 rho]
    ours_w1_st30        ours_w1 + reg_spike_loss_ratio_start_ep=30
    ours_w1_st60        ours_w1 + start_ep=60
    ours_ch_st30        ours_ch + start_ep=30
    ours_ch_st60        ours_ch + start_ep=60
    ours_layer_w1_st30  ours_layer_w1 + start_ep=30
    ours_layer_w1_st60  ours_layer_w1 + start_ep=60
                (lambda 는 0-기준 에폭 start_ep 끝까지 0 이고 그 다음 에폭부터 rho*L_task/R 로
                 한 번에 들어간다 -- 램프 없음. 기존 팔(start_ep=0)도 첫 에폭은 lambda=0 이다)

    [끝 구간 규제 해제 -- 원 방법과 전부 같고 마지막 K 에폭만 lambda=0. knob 은 rho (26-10-05)]
    ours_ch_end40       ours_ch + reg_spike_loss_ratio_end_ep=40 (310에폭이면 271~310 이 lambda=0)
    ours_ch_end30       ours_ch + end_ep=30

    [서로게이트 손잡이 baseline -- base 와 전부 같고 발화 함수의 서로게이트만 knob 으로 정한다 (26-10-10)]
    base_sg     knob 형식  ahf-w<폭>-h<높이>   -> 'asy_height_fix' (Surro 2025-11), surro_asyhf_width/height
                           asym-w<폭>-h<높이>  -> 'asym', surro_grad_alpha(폭)/surrogate_bias(높이 자리)
                예) python run_paper.py vggdvscm base_sg ahf-w1.0-h0.75 --seeds 1
                (VGGSNN-DVS 81~82% 를 낸 2025-11 런은 asy_height_fix, w=1.0, b=0.75)
    ours_ch_sg  ours_ch 와 전부 같고 서로게이트만 knob 으로 정한다 (26-10-10)
                knob 형식  <rho>@<서로게이트>  (서로게이트 문자열은 base_sg 와 같다)
                예) python run_paper.py vggdvscm ours_ch_sg 3e-3@ahf-w0.5-h0.75 --seeds 1
                디렉토리명은 '.'->'p' 만 바뀐다: vggdvscm-ours_ch_sg-3e-3@ahf-w0p5-h0p75-s1

**선별** — 조건당 5런을 돌리고 4개를 쓴다. 배제 규칙은 제안법과 비교군에 똑같이 적용한다:
    (1) 310에폭 미완주  (2) S30/S1 < 0.19  (3) 그래도 5개 남으면 val_acc 상위 4개.
판정은 `collect_paper.py` 가 한다.
"""
import argparse, math, os, re, subprocess, sys, threading, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 서버마다 conda 환경 위치가 다르다. EIP_PYTHON 으로 덮어쓴다.
#   EIP_PYTHON=/srv2/kyccj/miniconda3/envs/venv_1/bin/python python run_paper.py ...
PYTHON = os.environ.get('EIP_PYTHON', '/home/kyccj/anaconda3/envs/venv_1/bin/python')
CUDA_LD_PATH = os.environ.get(
    'EIP_CUDA_LD', os.path.join(os.path.dirname(os.path.dirname(PYTHON)), 'lib'))

# 조합 -> 원본 config (모델·데이터셋·학습 하이퍼파라미터를 그대로 물려받는다)
SOURCES = {
    'r19c10':   ('_rho_agg/agg_r19_c10',                 'ResNet19', 'CIFAR10'),
    'vggc10':   ('_sweep_wta_rev/lambda_1e-07',          'VGG16',    'CIFAR10'),
    'r19c100':  ('_sweep_wta_rev_r19_c100/lambda_1e-07', 'ResNet19', 'CIFAR100'),
    'vggc100':  ('_sweep_wta_rev_c100/lambda_1e-07',     'VGG16',    'CIFAR100'),
    # 26-09-29 VGGSNN x CIFAR10-DVS. 원본은 git 추적 경로(configs_paper/)에 둔다 -- 위 넷은
    # _*/ 라 .gitignore 대상이어서 sejong 이 pull 로 못 받는다. 레시피(200ep, batch 32, T=4)와
    # 근거는 그 파일 머리 주석에 있다. 기존 네 조합의 생성 config 는 이 줄과 무관하다.
    'vggdvs':   ('configs_paper/vggdvs',                 'VGGSNN',   'CIFAR10_DVS'),
    # 26-10-09 vggdvs 의 2025-11 레시피 복원판 (nda_cutmix + 워밍업 20). 근거는 그 파일 머리 주석.
    'vggdvscm': ('configs_paper/vggdvs_cm',              'VGGSNN',   'CIFAR10_DVS'),
    # 26-10-15 vggdvs 와 같은 원본 config + '첫 프레임' 입력 (프레임 0 을 T 번 반복해 모든 스텝에 넣는다.
    # 원본 Surro/Base 코드의 입력 방식). config 는 vggdvs 와 바이트 단위로 같고 main_sweep.py 만
    # 다르다 (아래 FRAME0). 근거: _ours/vggdvs-base-f0-s1 val 81.96 (n=1) vs vggdvs base 74.40/73.29 (n=2) (decisions.md 10-14).
    'vggdvsf0': ('configs_paper/vggdvs',                 'VGGSNN',   'CIFAR10_DVS'),
    # 26-10-15 vggdvscm 과 원본 config·main_sweep.py 가 같고 단계별 BN(bn_per_step=True)만 다르다
    # (아래 COMBO_FLAGS). 입력은 vggdvscm 그대로 스텝 t 에 프레임 t. 비교 대상 vggdvscm-base---s1~3.
    'vggdvscmtd': ('configs_paper/vggdvs_cm',            'VGGSNN',   'CIFAR10_DVS'),
    # 26-10-26 문헌 데이터 프로토콜 (UniSparse 2026 기준 전처리): T=10 이벤트 수 10등분 프레임, 48 bicubic+antialias,
    # 클립 단위 증강, 클래스별 900/100 (시험 1000장 = valid), T·B 묶음 BN. 모델·학습 레시피는 우리 것이다 (7축 차이는
    # configs_paper/vggdvs_lit/config_sweep.py 머리 주석) -- '문헌 설정'이라고 부르지 않는다. 손실은 스텝 출력을
    # 복원한 TET (lib_snn/tet_loss.py out='step'). 위 T=4 조합들과 섞지 않는다 (rho·s_count·R·val 의 성격이 다르다).
    # 아래 COMBO_FLAGS·LIT 와 lit 전용 새 파일 (datasets/cifar10_dvs_lit.py, lib_snn/tet_loss.py,
    # lib_snn/lit_spike_metrics.py, build_dvslit_cache.py) 로만 켜진다.
    # 원본 config 는 vggdvs 복사본에 T=10, batch 40, data_aug_mix='None' 만 바꾼 것 (그 파일 머리 주석).
    'vggdvslit': ('configs_paper/vggdvs_lit',             'VGGSNN',   'CIFAR10_DVS'),
    # 26-11-02 r19c10 과 원본 config 가 같고 time_step 만 8 (아래 COMBO_FLAGS, 사용자 승인 11-02).
    # 원본은 time_step 을 대입하지 않아 flags.py 기본값 4 로 돈다. T=4 조합들과 s_count·rho 눈금이 달라 섞지 않는다.
    'r19c10t8': ('_rho_agg/agg_r19_c10',                  'ResNet19', 'CIFAR10'),
}

# 26-10-15 조합에 묶인 플래그. 방법(METHODS)·BASE 가 아니라 조합에 붙여 방법과 무관하게 이 조합에서만 켠다.
# 사용자 지시(10-15): 단계별 BN 은 VGGSNN x CIFAR10_DVS 에서만, 다른 방법·조합에는 적용 금지.
# 여기 없는 조합은 덮어쓰기 블록이 이전과 바이트 단위로 같다 (bn_per_step 줄 자체가 안 생긴다. verify 가 확인).
# bn_per_step = BN 통계(평균·분산, 이동평균)는 T*B 에서, 정규화는 스텝별 (flags.py 주석).
COMBO_FLAGS = {'vggdvscmtd': {'bn_per_step': True},
               # 26-10-26 문헌 BN 은 T*B 묶음 BN (flatten(0,1)+BatchNorm2d) -- bn_per_step 이 바로 그 구조다.
               # 평균 BN 이면 전 프레임 입력의 시험 시 BN 입력이 '전체 히스토그램/T' 가 되어 분할 방식이 거의
               # 드러나지 않는다 (학습 시에는 증강이 있어 근사). 프레임 0 반복(vggdvsf0)은 이와 달리 분할 방식이
               # 입력 내용을 바꾼다.
               'vggdvslit': {'bn_per_step': True},
               # 26-11-02 R19-C10 T=8. 배치·에폭·학습률은 r19c10 그대로다.
               'r19c10t8': {'time_step': 8}}

# 26-10-15 '첫 프레임' 조합. 이 조합만 생성되는 main_sweep.py 의 데이터 로드 직후에 FRAME0_PATCH 를
# 끼운다 -- lib_snn/·datasets/ 는 안 건드린다. 패치 문자열은 10-14 수동 런 _ours/vggdvs-base-f0-s1 의
# main_sweep.py 에 들어간 것과 바이트 단위로 같다 (학습·검증·테스트 셋 모두에 적용, 로그에 [FRAME0]).
# 다른 조합의 main_sweep.py 는 이전과 같다 (make_main 참고).
FRAME0 = {'vggdvsf0'}
FRAME0_ANCHOR = '    datasets.datasets.load()\n'
FRAME0_PATCH = (
    "# >>> frame0 (26-10-14): 원본 Surro(InputGenLayer 가 inputs[:,0] 반환)와 같이 첫 프레임만 T번 반복 입력.\n"
    "# vggdvs(nda, 프레임별 독립 증강) 조건에서 입력 노드가 74 vs 82 차이의 원인인지 확정하는 런 (사용자 승인 10-14).\n"
    "def _f0(x, y):\n"
    "    return tf.tile(x[:, :1], [1, config.flags.time_step, 1, 1, 1]), y\n"
    "train_ds = train_ds.map(_f0); valid_ds = valid_ds.map(_f0); test_ds = test_ds.map(_f0)\n"
    "print('[FRAME0] input = frame 0 repeated T times', flush=True)\n"
    "# <<< frame0\n")

# 26-10-26 문헌 데이터 프로토콜 조합. 이 조합만 생성되는 main_sweep.py 에 세 패치를 끼운다 (FRAME0 와 같은 방식,
# lib_snn/·datasets/ 의 기존 파일은 안 건드린다). 기준 줄이 각각 정확히 1개여야 하고 아니면 투입 전에 멈춘다.
#   (1) 데이터: datasets.cifar10_dvs 를 datasets/cifar10_dvs_lit.py 로 바꿔 끼운다. datasets/datasets.py 가
#       호출 시점에 getattr(datasets, 'cifar10_dvs') 로 찾으므로 load() 가 새 로더로 간다.
#   (2) 모델 직후 (dist_strategy.scope() 안): 출력 [B,T,C] + TET 손실로 다시 compile (lib_snn/tet_loss.py).
#       out='step': n_fc 누적 막전위의 softmax 에서 스텝 출력 u_t (+클래스 무관 상수) 를 복원해 문헌식 손실·평가.
#   (3) 콜백 직후: 에폭 끝 GPU 메모리 peak 로그 (연기시험 실측용) + 문헌 정의 스파이크 집계 (n_in 제외 은닉
#       스파이크·발화율·SOP, lib_snn/lit_spike_metrics.py) + TET 1e-30 바닥 감시 (PFloorLogger).
#       셋 다 읽기만 한다 (학습 무영향, logs 키 추가 없음).
# 패치마다 로그 표식([DVSLIT] [TET] [MEM] [PFLOOR] [LITSPK])을 남긴다 -- 패치 없이 돈 런은 표식이 없어 원장에서 거른다.
# 다른 조합의 main_sweep.py 는 이전과 같다 (make_main 참고).
LIT = {'vggdvslit'}
LIT_PATCHES = (
    ('import datasets\nimport callbacks\n',
     "# >>> dvslit (26-10-26): 문헌 프로토콜 로더 (datasets/cifar10_dvs_lit.py)\n"
     "import datasets.cifar10_dvs_lit as _dvslit\n"
     "datasets.cifar10_dvs = _dvslit\n"
     "print('[DVSLIT] loader = datasets.cifar10_dvs_lit', flush=True)\n"
     "# <<< dvslit\n"),
    ('    model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch,valid_ds)\n',
     "    # >>> dvslit (26-10-26): 출력 [B,T,C] + TET 손실로 다시 compile (lib_snn/tet_loss.py)\n"
     "    import lib_snn.tet_loss as _tet\n"
     "    _tet.apply(model, config, out='step')\n"
     "    # <<< dvslit\n"),
    ('        callbacks.callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num)\n',
     "    callbacks_train.append(_tet.MemPeakLogger())    # dvslit: 에폭 끝 GPU 메모리 peak\n"
     "    callbacks_train.append(_tet.PFloorLogger())    # dvslit: TET 1e-30 바닥에 걸린 p 비율 (읽기 전용)\n"
     "    import lib_snn.lit_spike_metrics as _litspk\n"
     "    callbacks_train.append(_litspk.LitSpikeLogger(model, config))    # dvslit: 문헌 정의 스파이크 (n_in 제외)\n"),
)


def _load_lit():
    """datasets/cifar10_dvs_lit.py 를 파일 경로로 불러온다 (datasets/__init__ 이 config·TF 를 끌어오지 않게).
    그 모듈은 맨 위에서 numpy 만 import 한다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_cifar10_dvs_lit', os.path.join(PROJECT_ROOT, 'datasets', 'cifar10_dvs_lit.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# 모든 방법이 공유하는 바닥값. 켜고 끄는 것을 전부 명시해 원본 상태에 안 기댄다.
BASE = {
    'reg_spike_out': True,
    'reg_spike_log_detail': True,       # spike_count 는 reg 와 무관하게 찍히므로 L2 에서도 S30/S1 이 나온다
    'reg_spike_R_per_step': False,      # 26-08-07 이전 회계(R 을 T배 과소보고)와 섞지 않는다
    'reg_spike_accum_loss': False,
    'reg_spike_out_encourage': False,
    'reg_spike_starget': False,
    'reg_spike_adaptive': False,
    'reg_spike_grow': False,
    'reg_spike_sc_feedback': False,
    'reg_spike_lr_linked': False,
    'reg_spike_lr_brake': False,
    'sc_loss_scd': False,
    'reg_spike_wta_rev_w1': False,      # 기본 경로(실효 기울기 w^2). ours_w1 만 True 로 덮는다
    'reg_spike_maxnorm_no_inv': False,  # 1- 반전 유지. abl_noinv_wc 만 True 로 덮는다
}
WTA = {'reg_spike_out_sc': True, 'reg_spike_out_wta_rev': True}
MAXNORM = {'reg_spike_out_sc_maxnorm': True, 'reg_spike_maxnorm_group': "'within_channel'",
           'reg_spike_out_sc_maxnorm_plain': False, 'reg_spike_out_sm_plain': False,
           'reg_spike_out_sc_one': False}
VMEM = {'reg_spike_vmem_gain': 1.0, 'reg_spike_vmem_silent_only': True}
RATIO = lambda rho: {'reg_spike_loss_ratio': True, 'reg_spike_loss_ratio_target': rho,
                     'reg_spike_loss_ratio_start_ep': 0}
FIXED = lambda lmb: {'reg_spike_loss_ratio': False, 'reg_spike_out_const': lmb}


def SG(knob, who):
    """서로게이트 손잡이 'ahf-w<폭>-h<높이>' / 'asym-w<폭>-h<높이>' -> 덮어쓸 플래그 dict.
    base_sg 에 있던 파서를 26-10-10 ours_ch_sg 와 공유하려고 함수로 뺐다 (동작 동일).
    who 는 오류 메시지에 찍을 방법 이름."""
    m = re.fullmatch(r'(ahf|asym)-w([0-9.]+)-h([0-9.]+)', str(knob))
    if not m:
        raise SystemExit(f'{who} knob 형식 오류: {knob!r} (예: ahf-w1.0-h0.75, asym-w1.0-h0.6)')
    fn, w, h = m.group(1), float(m.group(2)), float(m.group(3))
    if fn == 'ahf':
        sg = {'fire_surro_grad_func': "'asy_height_fix'", 'surro_asyhf_width': w,
              'surro_asyhf_height': h}
    else:
        sg = {'fire_surro_grad_func': "'asym'", 'surro_grad_alpha': w, 'surrogate_bias': h}
    return sg


def METHODS(name, knob):
    """방법 이름과 손잡이 값 -> 덮어쓸 플래그 dict."""
    if name == 'base':
        # 규제 없음. knob 은 안 쓴다 (CLI 자리를 맞추려면 '-' 를 넘긴다).
        # reg 블록이 통째로 안 돌지만 list_spike_count 는 proc.py:1040 에서 따로 채워지므로
        # spike_count 와 S30/S1 은 그대로 나온다.
        return {**{k: v for k, v in BASE.items() if k != 'reg_spike_out'},
                'reg_spike_out': False, 'reg_spike_loss_ratio': False}
    if name == 'prop':
        return {**BASE, **WTA, **MAXNORM, **VMEM, 'reg_spike_final_step': True, **RATIO(knob)}
    if name == 'ours_w1':
        # prop 와 **모든 설정이 같고** reg_spike_wta_rev_w1 만 True.
        # wta_rev 의 backward 가 spike 에 sc_rate 를 두 번 곱해 실효 기울기가 w^2 였다
        # (곱셈이 custom_gradient 바깥에 있어서다. flags.py 의 해당 항목 참고).
        # 이 팔은 설계 의도대로 w 한 번일 때 무엇이 달라지는지를 묻는다.
        # forward(R)는 두 경로가 같으므로 loss-ratio 의 rho 눈금도 그대로다.
        return {**METHODS('prop', knob), 'reg_spike_wta_rev_w1': True}
    if name == 'ours_ch':
        # prop 와 전부 같고 max 를 잡는 범위만 다르다: 채널의 공간 합끼리 경쟁한다
        # (neurons.py 의 'channel' 분기). 채널 안 모든 위치가 같은 w 를 받으므로
        # 순수한 채널 선별기다 -- 약한 채널이 통째로 눌린다. within_channel 이 못 보는
        # 채널 전체 활동량을 본다. 구조적 프루닝 각도를 재는 팔이다.
        return {**METHODS('prop', knob), 'reg_spike_maxnorm_group': "'channel'"}
    if name in ('ours_ch_b7', 'ours_ch_b4'):
        # ours_ch 와 전부 같고 reg_spike_shape_beta 만 다르다 (sc_rate = 1 - beta*sc/max).
        # beta=1(기본)은 1등이 w=0 으로 완전 면제인데, channel 묶음에서는 그게 1등 **채널
        # 통째**라 예산의 최대 덩어리가 압력을 못 받는다 -- 09-19 중간값에서 스파이크가
        # 434K/391K/358K 로 거의 안 줄었다 (ours 는 같은 rho 에서 290K/234K/192K).
        # beta<1 이면 1등도 w=1-beta 를 받는다. loss-ratio 가 lambda 를 다시 풀어 주므로
        # 이 스윕은 lambda 스윕으로 위장되지 않는다 (그래서 shape_mean1 은 켜지 않는다).
        beta = 0.7 if name.endswith('b7') else 0.4
        return {**METHODS('ours_ch', knob), 'reg_spike_shape_beta': beta}
    if name == 'ours_ch_taylor':
        # 26-10-02 ours_ch 와 전부 같고 채널 서열의 기준만 다르다: 스파이크 공간합 대신
        # Taylor 중요도 imp_c = |sum_{i in c} (dL/ds_i)*s_i| (배치 평균, EMA). 벌점을 받는 양
        # ||s*w||_2 는 그대로 스파이크고, w = 1 - imp/max imp 로 1등 채널이 면제되는 것도 같다.
        # '많이 쏘는 채널' 대신 '손실에 기여하는 채널'을 보호하는지 묻는 팔이다.
        return {**METHODS('ours_ch', knob), 'reg_spike_ch_importance': "'taylor'"}
    if name == 'ours_chxwc':
        # 26-10-05 ours_ch 와 전부 같고 가중만 inter(채널 공간합/채널 max) 와 intra(채널 안 위치/
        # 위치 max) 의 곱이다 (neurons.py 'channel_x_within'). 두 성분 모두 같은 scp(vmem readiness
        # 포함)를 쓴다. 1등 채널 전체와 각 채널의 1등 위치가 w=0. w 가 전반적으로 작아져 R 이
        # 줄고 loss-ratio 가 lambda 를 키우므로 같은 rho 라도 ours_ch 와 착지가 다르다.
        # shape_mean1 은 켜지 않는다 (ours_ch_b* 와 같은 이유).
        return {**METHODS('ours_ch', knob), 'reg_spike_maxnorm_group': "'channel_x_within'"}
    if name == 'sm':
        return {**BASE, **WTA, 'reg_spike_out_sc_sm': True, 'reg_spike_out_sc_maxnorm': False,
                'reg_spike_final_step': False, **FIXED(knob)}
    if name == 'l2':
        # sc 경로를 끄면 plain L2 가지로 간다 (neurons.py:1238)
        return {**BASE, 'reg_spike_out_sc': False, 'reg_spike_out_norm': True,
                'reg_spike_out_norm_sq': False, 'reg_spike_final_step': False, **FIXED(knob)}
    if name == 'l2_lr':
        # l2 와 **모든 설정이 같고** lambda 만 loss-ratio 가 푼다. knob 은 lambda 가 아니라 rho.
        # "제안법은 lambda 를 손으로 안 찾아도 된다" 는 주장의 대조군이다 -- 제어기 자체의 몫과
        # 1-maxnorm/vmem/final_step 의 몫을 갈라준다. 이게 없으면 둘이 섞인 채로만 보인다.
        # 고정 lambda 자리는 지운다 (reg_spike_out_const 를 rho 로 남겨두면, 플래그가 잘못
        # 꺼졌을 때 rho 가 조용히 lambda 로 쓰인다).
        # 옛 가지가 제어기를 받게 된 경위는 lib_snn/neurons.py 의 해당 주석 참고.
        base = {k: v for k, v in METHODS('l2', knob).items() if k != 'reg_spike_out_const'}
        return {**base, **RATIO(knob)}
    if name == 'bpsr':
        # BPSR (Yan et al. 2022, Front. Neurosci.) eq.(4) 의 spiking sparsity 항 그대로.
        # (lambda/2) * sum_t sum_i s^2. 스파이크가 이진이라 사실상 lambda/2 x 총 스파이크 수이고
        # 발화 뉴런당 gradient 가 lambda 로 **상수**다. 우리 l2 가지(sqrt(sum s^2))는
        # 뉴런당 lambda/sqrt(N) 라 층마다 최대 30배, 학습 중 2배 달라져서 lambda 재조정으로
        # 맞출 수 없다. 문헌 표준 기준선을 인용하려면 이 팔이 따로 있어야 한다.
        return {**BASE, 'reg_spike_out_sc': False, 'reg_spike_out_norm': False,
                'reg_spike_out_norm_sq': False, 'reg_spike_out_bpsr': True,
                'reg_spike_final_step': False, **FIXED(knob)}
    if name == 'suetake':
        # 26-10-04 Suetake et al. (arXiv 2302.01500) 의 synaptic interaction penalty, 순수 대조군.
        # lambda * sum_l psi_l * sum_i S_i / B (p=1, S = T 합 스파이크, B 배치 평균),
        # psi_l = 층 l 뉴런 하나의 fan-out (lib_snn/suetake.py 가 학습 시작 때 그래프에서 계산).
        # lambda 는 knob 고정값에 원 논문의 에폭 비례 선형 램프 (epoch+1)/E 를 곱한다.
        # l2/bpsr 과 같은 패턴: sc 가지를 끄고 옛 가지 플래그(norm·norm_sq·bpsr)도 다 꺼서
        # neurons.py 의 suetake 가지로만 간다 (그 가지는 경쟁·maxnorm·vmem·wta_rev 를 안 읽는다).
        # final_step 은 끈다 -- p=1 은 매 타임스텝 psi*s_t 를 더해 합이 psi*S 가 된다.
        return {**BASE, 'reg_spike_out_sc': False, 'reg_spike_out_norm': False,
                'reg_spike_out_norm_sq': False, 'reg_spike_out_bpsr': False,
                'reg_spike_suetake': True, 'reg_spike_suetake_p': 1,
                'reg_spike_suetake_ramp': True,
                'reg_spike_final_step': False, **FIXED(knob)}
    if name in ('unisparse', 'unisparse_opt', 'atlif', 'atlif_regonly'):
        # 26-10-20 임계값 경로 대조군 (compare-candidates-261020.md 3.2 B, 구현 lib_snn/thr_reg.py).
        # base 와 전부 같다 -- reg_spike_out=False 라 neurons.py 의 기존 규제 가지는 하나도 안 돈다. 손실에도
        # 아무것도 안 더한다. 벌점 몫은 train_step 의 apply_gradients 앞뒤에서 더한다
        # (unisparse: 바로 앞 BN 의 gamma·beta, atlif: 층당 theta). lambda 는 두 원 논문과 같이 학습 내내 고정이고
        # 실제 폭은 우리 학습률 스케줄(워밍업 20에폭 + 코사인)을 따라간다.
        # *_opt / *_regonly 는 갱신 방식만 다른 착지 시험 비교판이다 (10-20 리뷰).
        # knob 은 e 표기만 받는다 (원장 merge_inv.py 의 이름 정규식이 [0-9p]+e-?[0-9]+ 라 그 밖이면 조용히 빠진다).
        # '1..2e-3' 같은 오타는 config import 때, '1e999'(inf) 는 수십 시간 뒤 NaN 으로 죽으므로 여기서 거른다.
        # 0 은 받는다 (과제 몫만 재는 시험용 '0e0').
        try:
            v = float(knob)
        except ValueError:
            v = float('nan')
        if not re.fullmatch(r'[0-9.]+e-?[0-9]+', str(knob)) or not math.isfinite(v) or v < 0:
            raise SystemExit(f'{name} knob 형식 오류: {knob!r} (0 이상 유한값, e 표기, 예: 1e-6)')
        if name.startswith('unisparse'):
            upd = "'optimizer'" if name == 'unisparse_opt' else "'direct'"
            return {**METHODS('base', '-'), 'reg_spike_unisparse': True, 'reg_spike_unisparse_lambda': knob,
                    'reg_spike_unisparse_update': upd}
        task = "'none'" if name == 'atlif_regonly' else "'sgd'"
        return {**METHODS('base', '-'), 'reg_spike_atlif': True, 'reg_spike_atlif_lambda': knob,
                'reg_spike_atlif_theta_task': task}
    if name == 'ours_layer':
        # 제안법에서 채널 내 max 를 **층 전체 max** 로 바꾼다 (maxnorm_group='none').
        # within_channel 은 채널마다 최고 뉴런이 sc_rate=0 으로 면제되어 어떤 채널도
        # 통째로 죽지 않는다 (측정: 채널 단위 완전침묵이 plain L2 대비 50개뿐).
        # 'none' 은 약한 채널을 통째로 지울 수 있어 구조적 프루닝에는 유리하지만
        # 정확도를 잃을 수 있다. 그 맞바꿈의 크기를 아직 모른다.
        return {**BASE, **WTA, **MAXNORM, **VMEM,
                'reg_spike_maxnorm_group': "'none'",
                'reg_spike_final_step': True, **RATIO(knob)}
    if name == 'ours_layer_w1':
        # ours_layer 와 **모든 설정이 같고** reg_spike_wta_rev_w1 만 True.
        # 기본 경로의 실효 기울기가 w^2 인 것은 설계가 아니라 결과다 -- neurons.py 가
        # x = spike*sc_rate 를 custom_gradient **바깥에서** 곱해, backward 가 이미 w 를
        # 실은 기울기에 w 가 한 번 더 붙는다. 설계 의도는 w 한 번이다.
        # 이 팔은 그 의도대로의 w^1 을 층 전체 경쟁(maxnorm_group='none')과 묶는다.
        # forward(R)는 두 경로가 같으므로 loss-ratio 의 rho 눈금이 ours_layer 와 공유된다
        # -- 같은 rho 끼리 바로 비교할 수 있고 rho 를 다시 찾을 필요가 없다.
        return {**METHODS('ours_layer', knob), 'reg_spike_wta_rev_w1': True}
    if name == 'abl_nofinal':
        return {**BASE, **WTA, **MAXNORM, **VMEM, 'reg_spike_final_step': False, **RATIO(knob)}
    if name == 'abl_novmem':
        return {**BASE, **WTA, **MAXNORM, 'reg_spike_vmem_gain': 0.0,
                'reg_spike_vmem_silent_only': False,
                'reg_spike_final_step': True, **RATIO(knob)}
    if name == 'abl_nolr':
        # loss-ratio 제거 — 제안법 그대로에 **고정 lambda**. knob 이 rho 가 아니라 lambda 다.
        # loss-ratio 는 lambda 를 매 에폭 rho*L_task/R 로 다시 푼다. 실측(prop rho=4e-3, n=5)
        # 궤적은 ep1 4.35e-7 -> ep50 4.9e-7 -> ep300 2.6e-7 로 올랐다가 반으로 떨어진다.
        # 시간평균 4.0e-7, 중반 고원 4.83e-7, 후반 2.58e-7.
        # 이 팔은 "그 궤적이 필요한가, 같은 착지점이면 그만인가" 를 묻는다.
        return {**BASE, **WTA, **MAXNORM, **VMEM, 'reg_spike_final_step': True, **FIXED(knob)}
    if name == 'abl_noinv_wc':
        # prop 와 **모든 설정이 같고** reg_spike_maxnorm_no_inv 만 True.
        # 1-maxnorm 가지를 그대로 타고 마지막의 sc_rate = 1 - beta*sc_norm 을
        # sc_rate = sc_norm 으로만 바꾼다. 따라서 within_channel 묶음과
        # vmem(silent_only, gain=1.0), final_step, loss-ratio 가 전부 살아 있다.
        # 기존 abl_noinv 는 maxnorm_plain 가지(neurons.py)로 가는데 그 가지는
        # reg_spike_maxnorm_group 과 reg_spike_vmem_gain 을 읽지 않는다 -- 즉 반전·채널 내
        # 묶음·vmem 세 요소 동시 제거라 단일 요소 어블레이션이 아니다.
        # 기존 abl_noinv 런의 재현이 거기 걸려 있어 그 팔은 그대로 둔다.
        return {**METHODS('prop', knob), 'reg_spike_maxnorm_no_inv': True}
    if name == 'abl_noinv':
        # 1- 반전 제거: sc_rate = sc/max. 침묵 뉴런이 sc_rate=0 이 되어 규제에서 빠진다
        return {**BASE, **WTA, **VMEM, 'reg_spike_out_sc_maxnorm': False,
                'reg_spike_out_sc_maxnorm_plain': True,
                'reg_spike_final_step': True, **RATIO(knob)}
    # --- ours_layer_w1 어블레이션 (09-26) ---------------------------------------------
    # 제안법이 09-23 에 prop 에서 ours_layer_w1 로 바뀌었는데 위 abl_* 는 전부 prop 에서
    # 뺀 것이라 어블레이션 표가 제안법과 안 맞는다. 아래 네 팔은 METHODS('ours_layer_w1')
    # 에서 시작해 **한 요소만** 덮어쓴다 (group='none', w1 은 그대로 남는다).
    # 위 abl_* 는 재현이 걸려 있어 건드리지 않는다.
    if name == 'lw1_novmem':
        # abl_novmem 과 같은 방식: gain=0 이면 neurons.py 가 readiness 를 아예 안 더한다.
        # silent_only 는 gain=0 에서 안 읽히지만 abl_novmem 과 맞춰 False 로 둔다.
        return {**METHODS('ours_layer_w1', knob), 'reg_spike_vmem_gain': 0.0,
                'reg_spike_vmem_silent_only': False}
    if name == 'lw1_nofinal':
        # 벌점 대상이 T 합(final_step)이 아니라 매 스텝 spike 가 된다 (neurons.py spike_reg).
        # w1 경로는 accum_loss 와만 배타이고 BASE 가 accum 을 끄므로 문제없다.
        return {**METHODS('ours_layer_w1', knob), 'reg_spike_final_step': False}
    if name == 'lw1_nolr':
        # loss-ratio 제거, 고정 lambda. knob 이 rho 가 아니라 **lambda** 다.
        # RATIO 가 넣은 키를 전부 지우고 FIXED 로 바꾼다 -- l2_lr 과 같은 이유로, 제어기
        # 키를 남겨두면 플래그 하나가 잘못 켜졌을 때 lambda 값이 rho 로 조용히 읽힌다.
        # 결과 키 집합은 abl_nolr 과 같은 모양이다 (target/start_ep 없음, out_const 있음).
        base = {k: v for k, v in METHODS('ours_layer_w1', knob).items()
                if k not in RATIO(knob)}
        return {**base, **FIXED(knob)}
    if name == 'lw1_noinv':
        # abl_noinv_wc 와 같은 플래그: 1-maxnorm 가지를 그대로 타고 sc_rate = sc_norm.
        # group='none'(층 전체 max)·vmem(silent_only)·final_step·loss-ratio·w1 은 유지된다.
        # neurons.py 확인 사항:
        #  - reg_spike_shape_beta 는 no_inv 분기에서 읽히지 않는다 (1- 가 없으면 beta 는
        #    단순 배율이라 loss-ratio 가 lambda 로 되푼다). 기본 1.0 그대로 둔다.
        #  - w1 경로(layers.l2_norm_wta_rev_w1)는 (spike_reg, sc_rate) 를 받아 안에서 곱할
        #    뿐 sc_rate 의 모양을 가정하지 않는다. no_inv 와 같이 켜도 forward 는
        #    ||spike*sc_norm||, spike 기울기는 sc_norm/||x|| (w^1) 이다. 배타 assert 는
        #    accum_loss·inv_s 두 개뿐이고 여기서는 둘 다 꺼져 있다.
        #  - 층 전체가 침묵하면 x 가 전부 0 이라 기울기 0 (condition 분기) -- 기존과 같다.
        #    침묵 뉴런은 vmem readiness 만큼 sc_norm in [0,1) 을 받아 정확히 0 은 아니다.
        #  - shape_mean1 은 켜지 않는다 (neurons.py 주석: no_inv 와 같이 켜면 배율이 튄다).
        return {**METHODS('ours_layer_w1', knob), 'reg_spike_maxnorm_no_inv': True}
    # --- ours_ch(=ours_inter_ch) 어블레이션 (26-10-04) ---------------------------------
    # lw1_* 와 같은 방식으로 METHODS('ours_ch') 에서 시작해 **한 요소만** 덮어쓴다.
    # (group='channel', w 는 기본 w^2 경로 -- ours_ch 가 prop 기반이라 wta_rev_w1=False 다.)
    if name == 'ch_novmem':
        # lw1_novmem/abl_novmem 과 같음: gain=0 이면 readiness 를 안 더한다. silent_only 는
        # gain=0 에서 안 읽히지만 두 기존 팔과 맞춰 False 로 둔다.
        return {**METHODS('ours_ch', knob), 'reg_spike_vmem_gain': 0.0,
                'reg_spike_vmem_silent_only': False}
    if name == 'ch_nofinal':
        # 벌점 대상이 T 합이 아니라 매 스텝 spike 가 되고, sc_rate 의 채널 합도 매 스텝의
        # 누적 spike_count 로 다시 계산된다 (abl_nofinal 과 같다).
        return {**METHODS('ours_ch', knob), 'reg_spike_final_step': False}
    if name == 'ch_noinv':
        # neurons.py 확인: no_inv 는 1-maxnorm 가지 끝(group 분기 뒤)에서 sc_rate = sc_norm 으로
        # 바꾼다. 'channel' 분기의 sc_norm = (채널 공간합)/max_c 을 채널 안에 broadcast 한 것이라
        # 반전만 빠지고 채널 묶음·vmem readiness(scp 에 들어감)·final_step·loss-ratio 는 그대로다.
        # maxnorm_plain(abl_noinv) 은 group 을 안 읽어 쓰지 않는다 (decisions.md 09-20).
        # 1등 채널이 w=1 로 가장 세게 눌리고, 침묵 채널은 readiness 합만큼만 받는다.
        return {**METHODS('ours_ch', knob), 'reg_spike_maxnorm_no_inv': True}
    if name == 'ch_nolr':
        # loss-ratio 제거, 고정 lambda. knob 이 **lambda**. lw1_nolr 과 같은 방식으로 RATIO 키를
        # 전부 지우고 FIXED 로 바꾼다 (키 집합은 abl_nolr 과 같은 모양).
        base = {k: v for k, v in METHODS('ours_ch', knob).items() if k not in RATIO(knob)}
        return {**base, **FIXED(knob)}
    if name == 'ch_nowtarev':
        # 26-10-28 '- wta_rev' (사용자 승인 10-28). ours_inter_ch 에서 wta_rev backward 만 끈다.
        # 키는 ours_ch 어블레이션 관례 ch_* 를 따른다 (10-28 abl_nowtarev 에서 이름만 바꿈. 정의는 같고,
        # 옛 이름으로 돈 것은 연기시험 _smoke/r19c10-abl_nowtarev-3e-3-s901 하나뿐이다).
        # neurons.py 의 sc 가지는 그대로 돈다: sc_rate(1-maxnorm, channel 묶음, vmem readiness),
        # spike_reg(final_step 의 T 합), sc_loss = spike_reg*sc_rate 까지 같고, 마지막 노름만
        # l2_norm_wta_rev 대신 `elif conf.reg_spike_out_norm` 의 l2_norm 으로 간다. 두 함수 forward 는
        # 같은 sqrt(sum(x^2)) 라 R·sc_loss_snap·loss-ratio 의 rho 눈금이 ours_ch 와 공유된다.
        # 기울기: wta_rev 는 dL/dS = lambda*w^2/R (S=0 인 쌍에도), 여기는 자동미분 lambda*w^2*S/R.
        # sc_rate 계산 쪽에서 wta_rev 를 읽는 elif(neurons.py 1118)는 maxnorm 가지 뒤라 안 닿는다.
        # wta_rev 가 꺼지면 out_norm/out_norm_sq 가 노름 가지를 정하므로 원본 config 에 안 기대고
        # 명시한다 (SOURCES 원본은 모두 out_norm=True, norm_sq 는 기본 False -- 값은 그대로).
        # wta_rev_w1 은 BASE 가 False 로 둔다 (True 면 neurons.py assert 가 막는다).
        return {**METHODS('ours_ch', knob), 'reg_spike_out_wta_rev': False,
                'reg_spike_out_norm': True, 'reg_spike_out_norm_sq': False}
    # --- 규제 시작 지연 (09-30) --------------------------------------------------------
    # 원 방법과 **모든 설정이 같고** reg_spike_loss_ratio_start_ep 만 30/60 으로 덮는다.
    # proc.py 의 loss-ratio 갱신(spike_count_epoch_end)이 확인한 동작:
    #  - 에폭 끝(Keras 0-기준 epoch)마다 다음 에폭의 lambda 를 정한다. epoch < start_ep 이면 0.
    #    초기값도 0 (model.py adaptive_lambda) 이라 0-기준 0..start_ep 에폭은 lambda=0,
    #    0-기준 start_ep+1 (로그의 'Epoch start_ep+2/310') 부터 규제가 걸린다.
    #    기존 팔(start_ep=0)의 첫 규제 에폭이 'Epoch 2' 이므로 정확히 start_ep 에폭 늦는다.
    #  - 켜질 때는 lambda = rho*L_task/R 로 한 번에 점프한다 (램프 없음). 완화 장치는
    #    reg_spike_lr_brake(성장 상한 포함)뿐인데 BASE 가 끈다.
    #  - lambda=0 인 동안에도 sc_loss_snap(R)은 lambda 곱하기 전 값으로 찍히고 reg CSV·reg_R
    #    로그도 그대로 나온다. add_loss 는 0*R 이라 기울기가 없다.
    if name in ('ours_w1_st30', 'ours_w1_st60', 'ours_ch_st30', 'ours_ch_st60',
                'ours_layer_w1_st30', 'ours_layer_w1_st60'):
        src, st = name.rsplit('_st', 1)
        return {**METHODS(src, knob), 'reg_spike_loss_ratio_start_ep': int(st)}
    # --- 끝 구간 규제 해제 (26-10-05) ---------------------------------------------------
    # start_ep 의 거울: 원 방법과 **모든 설정이 같고** 마지막 K 에폭만 lambda=0 (proc.py).
    # 에폭 끝에 다음 에폭 lambda 를 정하므로 proc.py 는 0-기준 epoch >= E-K-1 에서 0 을 넣고,
    # 그 결과 로그의 'Epoch E-K+1/E' ~ 'Epoch E/E' 정확히 K 에폭이 무규제다.
    # 마지막 에폭 끝에 찍히는 adp_lambda 는 쓰이지 않는 값이다 (start_ep 의 첫 에폭과 대칭).
    if name in ('ours_ch_end40', 'ours_ch_end30'):
        src, k = name.rsplit('_end', 1)
        return {**METHODS(src, knob), 'reg_spike_loss_ratio_end_ep': int(k)}
    if name == 'base_sg':
        # 26-10-10 규제 없는 baseline 에서 서로게이트만 바꾼다. 원본 config 의 'asym' 대입은
        # config.set() 앞 덮어쓰기 블록이 이긴다. 폭·높이를 float 로 넣어 verify() 의 repr 비교와 맞춘다.
        return {**METHODS('base', '-'), **SG(knob, 'base_sg')}
    if name == 'ours_ch_sg':
        # 26-10-10 ours_ch(=ours_inter_ch) 와 전부 같고 서로게이트만 knob 으로 정한다.
        # knob 형식 '<rho>@<서로게이트>' 예) 3e-3@ahf-w0.5-h0.75 -- '@' 앞은 ours_ch 의 rho,
        # 뒤는 base_sg 와 같은 서로게이트 문자열 (파서 SG 를 공유한다).
        rho, sep, sg = str(knob).partition('@')
        if not sep or not re.fullmatch(r'[0-9.]+e-?[0-9]+', rho):
            raise SystemExit(f'ours_ch_sg knob 형식 오류: {knob!r} (예: 3e-3@ahf-w0.5-h0.75)')
        return {**METHODS('ours_ch', rho), **SG(sg, 'ours_ch_sg')}
    raise SystemExit(f'모르는 방법: {name}')


# 실험 산출물(체크포인트·CSV·train.log)은 로컬 SSD 가 아니라 HDD 에 쌓는다.
# 루트 디렉토리가 652개까지 늘고 SSD 가 84% 차서 26-09-14 에 옮겼다.
# paper/ 는 논문 후보, archive/ 는 과거 실험 라인.
# 서버마다 저장 위치가 다를 수 있어 환경변수로 덮어쓸 수 있게 둔다.
#   EIP_STORE=/data/kyccj/EIP/paper python run_paper.py ...
STORE = os.environ.get('EIP_STORE', '/media/hdd1/kyccj/EIP/paper')

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_SET = re.compile(r'^config\.set\(\)$', re.M)   # \s*$ 로 두면 끝의 개행까지 먹는다
MARK_A, MARK_B = '# >>> run_paper overrides', '# <<< run_paper overrides'


def fmt(v):
    return v if isinstance(v, str) else repr(v)


def run_flags(combo, method, knob):
    """덮어쓸 플래그 전체 = 방법(METHODS) + 조합(COMBO_FLAGS). 26-10-15.
    COMBO_FLAGS 에 없는 조합이면 METHODS 와 같은 dict 다 (생성 config 가 이전과 같다).
    방법과 조합이 같은 플래그를 정하면 어느 쪽이 이기는지 애매하므로 멈춘다."""
    want = METHODS(method, knob)
    cf = COMBO_FLAGS.get(combo, {})
    both = set(want) & set(cf)
    if both:
        raise RuntimeError(f'{combo} {method}: 조합 플래그와 방법 플래그가 겹침 {sorted(both)}')
    return {**want, **cf}


def override_block(flags_d, seed, tag, epochs=0, fix_seed=False):
    lines = [MARK_A]
    # config_snn_training.py:35 의 `conf.root_model_save=conf.exp_set_name` 을 덮어쓴다.
    # 리터럴 tag 를 쓰는 이유: 같은 파일 L171 이 exp_set_name 에 '_asym' 을 덧붙이므로
    # conf.exp_set_name 을 참조하면 이 블록의 실행 순서에 결과가 달라진다.
    lines.append(f"conf.root_model_save = {STORE + '/' + tag!r}")
    for k in sorted(flags_d):
        lines.append(f'conf.{k} = {fmt(flags_d[k])}')
    # 시드는 기본적으로 **태그에만** 쓰고 RNG 는 안 건드린다 (run_seed=-1).
    #
    # `tf.keras.utils.set_random_seed` 로 전역 시드를 고정했더니 R19-C10 baseline 이
    # 96.65% -> 95.72% 로 0.93%p 떨어지고 런 간 sd 가 0.050 -> 0.429 로 8배가 됐다.
    # train-val 격차 부호가 뒤집힌 것이 원인을 말해 준다:
    #     기존(시드 없음)  train 95.05 / val 96.17  ->  -1.12  (증강이 세서 정상)
    #     시드 고정        train 95.85 / val 95.12  ->  +0.73  (증강이 쉬워짐)
    # 전역 시드를 박으면 tf.data 파이프라인의 난수 연산이 에폭마다 같은 시퀀스를
    # 재생해 증강 다양성이 사라진다. 후반부에 과적합한다.
    #
    # 게다가 시드를 고정해도 GPU 연산이 비결정적이라 재현도 안 된다 (같은 시드 2회:
    # val_acc 37.35% vs 41.31%). 얻는 것 없이 잃기만 하므로 쓰지 않는다.
    lines.append(f'conf.run_seed = {seed if fix_seed else -1}')
    if epochs:
        # 연기 시험용. 설정이 죽지 않는지 2에폭으로 먼저 본다 (계획 §7-1)
        lines.append(f'conf.train_epoch = {epochs}')
    lines.append(MARK_B)
    return '\n'.join(lines)


def src_text(combo):
    src = SOURCES[combo][0]
    return open(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')).read()


def make_main(combo):
    """런 디렉토리의 main_sweep.py 내용. FRAME0 조합이 아니면 26-10-15 이전과 같은 문자열이다."""
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    m = m.replace('from config_snn_training import config', 'from config_sweep import config')
    if combo in FRAME0:
        # 기준 줄이 정확히 하나여야 한다. main_snn_training.py 가 바뀌어 못 찾으면 투입 전에 멈춘다
        # (조용히 평균 프레임으로 도는 것을 막는다).
        n = m.count(FRAME0_ANCHOR)
        if n != 1:
            raise RuntimeError(f'{combo}: main_snn_training.py 의 데이터 로드 줄이 {n}개 (1개여야 함)')
        m = m.replace(FRAME0_ANCHOR, FRAME0_ANCHOR + FRAME0_PATCH)
    if combo in LIT:
        # FRAME0 와 같은 이유로 기준 줄이 각각 정확히 하나여야 한다 (조용히 옛 로더·CE 로 도는 것을 막는다)
        for anchor, patch in LIT_PATCHES:
            n = m.count(anchor)
            if n != 1:
                raise RuntimeError(f'{combo}: main_snn_training.py 의 기준 줄 {anchor.strip()!r} 이 {n}개 (1개여야 함)')
            m = m.replace(anchor, anchor + patch)
    return m


def active_name(txt):
    """주석이 아니라 실제로 마지막에 대입되는 exp_set_name.
    (08-21 에 주석 줄을 치환해 원본 체크포인트를 날린 사고 재발 방지)"""
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 대입이 없음')
    return ms[-1].split("'")[1]


def make_config(combo, method, knob, seed, gpu, tag, epochs=0, fix_seed=False):
    c = src_text(combo)
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu}"', c)
    if n != 1:
        raise RuntimeError(f'{tag}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='{tag}'", c)
    if n < 1:
        raise RuntimeError(f'{tag}: 활성 exp_set_name 없음')
    blk = override_block(run_flags(combo, method, knob), seed, tag, epochs, fix_seed)
    c, n = RE_SET.subn(blk + '\nconfig.set()', c, count=1)
    if n != 1:
        raise RuntimeError(f'{tag}: config.set() 을 못 찾음 ({n}개)')
    return c


def verify(path, combo, method, knob, seed, gpu, tag, epochs=0, fix_seed=False):
    a = open(path).read()
    b = src_text(combo)
    # 덮어쓰기 블록 / GPU / 이름 을 지운 나머지가 원본과 같아야 한다
    stripped = re.sub(re.escape(MARK_A) + r'.*?' + re.escape(MARK_B) + r'\n', '', a, flags=re.S)
    norm = lambda s: RE_NAME.sub('', RE_GPU.sub('', s))
    if norm(stripped) != norm(b):
        raise RuntimeError(f'{tag}: 덮어쓰기 블록/GPU/이름 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{tag}: GPU 지정 안 됨')
    if active_name(a) != tag:
        raise RuntimeError(f'{tag}: 유효 exp_set_name 이 {active_name(a)!r} — 원본 덮어쓸 위험')
    if active_name(b) == tag:
        raise RuntimeError(f'{tag}: 원본과 저장 경로가 같음')
    # 블록이 config.set() 앞에 있어야 효력이 있다
    if a.index(MARK_B) > a.rindex('config.set()'):
        raise RuntimeError(f'{tag}: 덮어쓰기 블록이 config.set() 뒤에 있음')
    # 요청한 플래그가 그대로 들어갔는지 (26-10-15 부터 조합 플래그 COMBO_FLAGS 포함)
    want = run_flags(combo, method, knob)
    for k, v in want.items():
        if not re.search(rf'^conf\.{re.escape(k)} = {re.escape(fmt(v))}$', a, re.M):
            raise RuntimeError(f'{tag}: {k}={fmt(v)} 가 블록에 없음')
    # 조합 플래그는 그 조합 밖의 config 에 run_paper 가 한 줄도 더하면 안 된다 (단계별 BN 범위 한정, 사용자 10-15).
    # 26-11-02 time_step 만 예외: vggdvs 계열 원본이 직접 대입하므로 '원본과 대입 줄 수가 같음' 으로 본다.
    # 나머지 키(bn_per_step)는 이전 그대로 원본에 있어도 막는다 (_*/ 원본은 gitignore 라 서버마다 따로 있다).
    for k in {k for cf in COMBO_FLAGS.values() for k in cf} - set(want):
        pat = rf'^\s*conf\.{re.escape(k)}\s*='
        allowed = len(re.findall(pat, b, re.M)) if k == 'time_step' else 0
        if len(re.findall(pat, a, re.M)) != allowed:
            raise RuntimeError(f'{tag}: {combo} 조합인데 {k} 대입이 config 에 있음 (원본 허용 {allowed}줄)')
    if want.get('bn_per_step') and SOURCES[combo][1:] != ('VGGSNN', 'CIFAR10_DVS'):
        raise RuntimeError(f'{tag}: bn_per_step 은 VGGSNN x CIFAR10_DVS 전용')
    want_seed = seed if fix_seed else -1
    if not re.search(rf'^conf\.run_seed = {want_seed}$', a, re.M):
        raise RuntimeError(f'{tag}: run_seed={want_seed} 설정 안 됨')
    # 모델·데이터셋이 요청한 조합인지
    _, model, dataset = SOURCES[combo]
    if not re.search(rf"^conf\.model\s*=\s*'{model}'", a, re.M):
        raise RuntimeError(f'{tag}: model 이 {model} 이 아님')
    if not re.search(rf"^conf\.dataset\s*=\s*'{dataset}'", a, re.M):
        raise RuntimeError(f'{tag}: dataset 이 {dataset} 이 아님')
    # 배타 조합
    if want.get('reg_spike_final_step') and want.get('reg_spike_accum_loss'):
        raise RuntimeError(f'{tag}: final_step 과 accum 을 같이 켤 수 없음')
    if want.get('reg_spike_loss_ratio_end_ep', 0) and not want.get('reg_spike_loss_ratio'):
        # proc.preproc 의 assert 와 같은 조건을 투입 전에 본다 (26-10-05)
        raise RuntimeError(f'{tag}: end_ep 는 loss-ratio 경로에서만 동작함')


def preflight(jobs, sweep_dir, epochs=0, fix_seed=False):
    bad = []
    for combo, method, knob, seed, tag in jobs:
        if combo not in SOURCES:
            bad.append(f'모르는 조합 {combo}')
            continue
        if not os.path.exists(os.path.join(PROJECT_ROOT, SOURCES[combo][0], 'config_sweep.py')):
            bad.append(f'{SOURCES[combo][0]} (원본 config 없음)')
        tgt = os.path.join(PROJECT_ROOT, tag)
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(f'{tgt} (체크포인트가 이미 있음)')
        if os.path.exists(os.path.join(sweep_dir, tag, 'train.log')):
            bad.append(f'{sweep_dir}/{tag}/train.log (이미 돈 흔적)')
    fl = open(os.path.join(PROJECT_ROOT, 'flags.py')).read()
    nr = open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()
    for tok in ('reg_spike_final_step', 'reg_spike_vmem_silent_only', 'run_seed'):
        if tok not in fl:
            bad.append(f'flags.py 에 {tok} 없음')
    for tok in ('reg_spike_final_step', 'reg_spike_vmem_silent_only'):
        if tok not in nr:
            bad.append(f'lib_snn/neurons.py 에 {tok} 반영 안 됨')
    # 26-10-15 단계별 BN 조합: 플래그 정의·적용·범위 가드가 이 작업 트리에 다 있어야 한다
    # (어느 하나라도 옛 파일이면 조용히 평균 BN 으로 돌거나 flags 에서 죽는다)
    if any(COMBO_FLAGS.get(j[0], {}).get('bn_per_step') for j in jobs):
        for rel in ('flags.py', 'models/vggsnn.py', 'lib_snn/model_builder.py'):
            if 'bn_per_step' not in open(os.path.join(PROJECT_ROOT, rel)).read():
                bad.append(f'{rel} 에 bn_per_step 반영 안 됨')
    # 26-10-20 임계값 경로 대조군: 플래그 정의·뉴런 분기·옵티마이저 뒤 갱신·에폭 훅이 이 작업 트리에 다 있어야
    # 한다 (옛 파일이 섞이면 갱신이 조용히 빠지거나 flags 에서 죽는다)
    for key in ('reg_spike_unisparse', 'reg_spike_atlif'):
        if any(run_flags(j[0], j[1], j[2]).get(key) for j in jobs):
            for rel in ('flags.py', 'lib_snn/neurons.py', 'lib_snn/model.py', 'lib_snn/proc.py'):
                if key not in open(os.path.join(PROJECT_ROOT, rel)).read():
                    bad.append(f'{rel} 에 {key} 반영 안 됨')
            if 'route_grads' not in open(os.path.join(PROJECT_ROOT, 'lib_snn', 'model.py')).read():
                bad.append('lib_snn/model.py 에 thr_reg.route_grads 반영 안 됨 (theta 가 AdamW 로 간다)')
            if not os.path.exists(os.path.join(PROJECT_ROOT, 'lib_snn', 'thr_reg.py')):
                bad.append('lib_snn/thr_reg.py 없음')
            if key == 'reg_spike_unisparse' and \
                    '_thr_stash' not in open(os.path.join(PROJECT_ROOT, 'lib_snn', 'layers.py')).read():
                bad.append('lib_snn/layers.py 에 BN 입력 참조(_thr_stash) 반영 안 됨')
    # 26-10-26 문헌 데이터 프로토콜 조합: 새 파일과 이 서버의 캐시(manifest 키·표본 수·파일 sha256 = canus 기준)가
    # 있어야 한다. 캐시 위치는 env EIP_DVSLIT_CACHE, 없으면 dirname(EIP_STORE)/dvslit_cache/<키> (로더와 같은 함수).
    # sha256 대조는 1.84GB 를 읽는다 (canus 약 9초). 다른 TF 로 다시 빌드한 원격 캐시를 여기서 거른다.
    if any(j[0] in LIT for j in jobs):
        for rel in ('datasets/cifar10_dvs_lit.py', 'lib_snn/tet_loss.py', 'lib_snn/lit_spike_metrics.py',
                    'build_dvslit_cache.py'):
            if not os.path.exists(os.path.join(PROJECT_ROOT, rel)):
                bad.append(f'{rel} 없음')
        if not bad:
            try:
                lit = _load_lit()
                lit.check_cache(lit.cache_dir(), verify_sha=True)
            except Exception as e:
                bad.append(f'dvslit 캐시: {e}')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    os.makedirs(sweep_dir, exist_ok=True)
    for combo, method, knob, seed, tag in jobs:
        tmp = os.path.join(sweep_dir, f'.pre_{tag}.py')
        open(tmp, 'w').write(make_config(combo, method, knob, seed, 0, tag, epochs, fix_seed))
        try:
            verify(tmp, combo, method, knob, seed, 0, tag, epochs, fix_seed)
        finally:
            os.remove(tmp)
        make_main(combo)    # FRAME0·LIT 조합의 기준 줄 확인 (파일은 안 쓴다)
    print(f'사전 점검 통과: {len(jobs)}개 config 생성·검증 완료', flush=True)


def run_one(gpu, combo, method, knob, seed, tag, sweep_dir, epochs=0, fix_seed=False):
    d = os.path.join(sweep_dir, tag)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(combo, method, knob, seed, gpu, tag, epochs, fix_seed))
    verify(cfg, combo, method, knob, seed, gpu, tag, epochs, fix_seed)
    open(os.path.join(d, 'main_sweep.py'), 'w').write(make_main(combo))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {tag}  "
          f"({combo} {method} knob={knob} seed={seed})", flush=True)
    # train.log 도 HDD 에 쌓는다 (런당 ~25MB, 누적 14GB 였다). 로컬에는 심볼릭 링크만
    # 남겨 collect_paper.py 등 _paper/<tag>/train.log 를 읽는 도구가 그대로 동작하게 한다.
    store_dir = os.path.join(STORE, tag)
    os.makedirs(store_dir, exist_ok=True)
    # 어느 서버에서 돌았는지 런 옆에 남긴다. canus(…138) 와 sejong(…23) 은 TF·numpy·
    # CUDA 버전이 달라서, 결과를 합칠 때 서버를 공변량으로 넣어야 한다.
    with open(os.path.join(store_dir, 'host.txt'), 'w') as hf:
        import socket as _s
        hf.write(f'{_s.gethostname()}\t{_s.gethostbyname(_s.gethostname())}\t'
                 f'gpu{gpu}\t{time.strftime("%Y-%m-%d %H:%M")}\n')
    log_path = os.path.join(store_dir, 'train.log')
    link = os.path.join(d, 'train.log')
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(log_path, link)
    with open(log_path, 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] DONE  {tag} rc={rc}", flush=True)
    return rc


def free_gpus(allowed, exclude, max_mib=200):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            timeout=60).decode()
    except Exception:
        return []
    res = []
    for line in out.strip().split('\n'):
        idx, mem = [x.strip() for x in line.split(',')]
        idx = int(idx)
        if idx not in allowed or idx in exclude:
            continue
        if int(mem) <= max_mib:
            res.append(idx)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('combo', nargs='?', help=' / '.join(SOURCES))
    ap.add_argument('method', nargs='?', help='base prop ours_w1 ours_ch ours_ch_b7 ours_ch_b4 ours_ch_taylor ours_chxwc l2 l2_lr sm bpsr suetake unisparse unisparse_opt atlif atlif_regonly ours_layer ours_layer_w1 abl_nofinal abl_novmem abl_noinv abl_noinv_wc abl_nolr lw1_novmem lw1_nofinal lw1_nolr lw1_noinv ch_novmem ch_nofinal ch_noinv ch_nolr ch_nowtarev ours_w1_st30 ours_w1_st60 ours_ch_st30 ours_ch_st60 ours_layer_w1_st30 ours_layer_w1_st60 ours_ch_end40 ours_ch_end30')
    ap.add_argument('knob', nargs='?', help="prop/ours_w1/ours_ch/ours_ch_taylor/ours_chxwc/ours_layer/ours_layer_w1/*_st30·*_st60/*_end40·*_end30/l2_lr/abl(nolr 제외)/lw1(nolr 제외)/ch_*(nolr 제외) 은 rho, sm/l2/bpsr/suetake/unisparse*/atlif*/abl_nolr/lw1_nolr/ch_nolr 은 lambda, base 는 '-'")
    ap.add_argument('--seeds', default='1,2,3,4,5', help='복제 시드 (쉼표). 서로 달라야 한다')
    ap.add_argument('--gpus', default='0,1,2,3,4,5', help='쓸 GPU (쉼표). 6·7 은 기본 제외')
    ap.add_argument('--dir', default='_paper', help='스윕 디렉토리')
    ap.add_argument('--list', action='store_true', help='조합·방법 목록만 출력')
    ap.add_argument('--dry', action='store_true', help='config 만 만들어 검증하고 끝')
    ap.add_argument('--epochs', type=int, default=0, help='연기 시험용 에폭 수 (0 = 원본 그대로)')
    ap.add_argument('--fix-seed', action='store_true', default=False,
                    help='RNG 를 실제로 고정한다. 기본은 끔 — 고정하면 tf.data 증강이 '
                         '에폭마다 같은 시퀀스를 재생해 R19-C10 이 96.65%%->95.72%% 로 떨어졌다')
    a = ap.parse_args()

    if a.list or not (a.combo and a.method and a.knob):
        print('조합:')
        for k, (s, m, d) in SOURCES.items():
            print(f'  {k:9s} {m:9s} {d:9s}  <- {s}')
        print('\n방법: base prop ours_w1 ours_ch ours_ch_b7 ours_ch_b4 ours_ch_taylor ours_chxwc l2 l2_lr sm bpsr suetake unisparse unisparse_opt atlif atlif_regonly ours_layer ours_layer_w1 abl_nofinal abl_novmem abl_noinv abl_noinv_wc abl_nolr lw1_novmem lw1_nofinal lw1_nolr lw1_noinv ch_novmem ch_nofinal ch_noinv ch_nolr ch_nowtarev ours_w1_st30 ours_w1_st60 ours_ch_st30 ours_ch_st60 ours_layer_w1_st30 ours_layer_w1_st60 ours_ch_end40 ours_ch_end30')
        print('\n예) python run_paper.py r19c10 prop 1e-3 --seeds 1,2,3,4,5 --gpus 0,2')
        return

    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    if len(set(seeds)) != len(seeds):
        raise SystemExit('시드가 중복이다 — 같은 시드면 같은 런이 나온다')
    allowed = {int(g) for g in a.gpus.split(',') if g.strip()}
    sweep_dir = os.path.join(PROJECT_ROOT, a.dir)

    knob_tag = str(a.knob).replace('.', 'p')
    jobs = [(a.combo, a.method, a.knob, s, f'{a.combo}-{a.method}-{knob_tag}-s{s}') for s in seeds]

    preflight(jobs, sweep_dir, a.epochs, a.fix_seed)
    if a.dry:
        print('--dry: 검증만 하고 종료')
        return

    pending, busy, threads = list(jobs), set(), []
    failed = []
    print(f'--- {a.combo} / {a.method} / knob={a.knob} : {len(pending)} runs '
          f'(GPU {sorted(allowed)}) ---', flush=True)
    while pending:
        for g in free_gpus(allowed, busy):
            if not pending:
                break
            combo, method, knob, seed, tag = pending.pop(0)
            busy.add(g)

            def worker(g=g, combo=combo, method=method, knob=knob, seed=seed, tag=tag):
                try:
                    rc = run_one(g, combo, method, knob, seed, tag, sweep_dir,
                                 a.epochs, a.fix_seed)
                    if rc:
                        failed.append((tag, rc))
                except Exception as e:              # 한 job 이 죽어도 나머지는 계속
                    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {g}] FAIL {tag}: {e}",
                          flush=True)
                    failed.append((tag, 'exception'))
                finally:
                    busy.discard(g)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
            time.sleep(90)          # 같은 GPU 를 두 job 이 잡는 것 방지
        if pending:
            time.sleep(60)
    for t in threads:
        t.join()
    if failed:
        # 학습이 죽었는데 0 을 반환하면 큐 실행기가 'done' 으로 표시한다.
        # sejong 에서 Keras 2.11 이 체크포인트 디렉토리를 안 만들어 6런이 9분 만에
        # done 으로 찍혔다 (26-09-14). 실패는 반드시 종료코드로 올린다.
        for tag, rc in failed:
            print(f'실패: {tag} rc={rc}', flush=True)
        sys.exit(1)
    print('--- 전부 완료 ---', flush=True)


if __name__ == '__main__':
    main()
