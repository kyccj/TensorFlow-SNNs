'''
    Configuration for SNN direct training

    run_paper_vgglit.py 의 vggdvslitlk5 조합 원본 (26-11-03, VGGSNN baseline 레시피 선별, 사용자 결정 11-03 '누설 0.5 만').
    configs_paper/vggdvs_lit/config_sweep.py (vggdvslit) 를 복사해 아래 값만 바꿨다. exp_set_name 은 그 줄을 고쳤고,
    누설은 원래 줄 바로 뒤에 덮어쓰는 줄을 넣었다. 나머지는 한 글자도 같다:
      exp_set_name 'EIP-SNN-26_vggdvslit-src' -> 'EIP-SNN-26_vggdvslitlk5-src'
      leak_const_init 0.9 -> 0.5   (n_reset_type 'reset_by_sub' 은 그대로)
      (최적화기·에폭은 vggdvslit 그대로: AdamW 6e-3, 감쇠 2e-2, 워밍업 10, 200에폭)
    neurons.py 는 적분 뒤 감쇠라 u_t = a (u_{t-1} + I_t) 이고, a=0.5 는 문헌 LIF tau=2 (decay_input) 의 충전식과 같다.
    리셋은 우리 것 그대로 (발화하면 Vth 를 뺀다, 기울기 1 - surrogate). 문헌은 0 으로 리셋 + detach_reset 이라 여기는 다르다.
    은닉 8층과 n_fc (출력층) 모두 a=0.5. TET out='step' 복원식은 n_fc 의 leak_const 를 모델에서 읽는다
    (lib_snn/tet_loss.py out_leak, [TET] 줄 'log p_t/0.5').
    vggdvslit 런과 같은 표에 섞지 않는다. 아래는 vggdvslit 원본의 머리 주석 그대로다.

    run_paper.py 의 vggdvslit 조합 원본 (VGGSNN x CIFAR10-DVS, 문헌 데이터 프로토콜 T=10, 26-10-26).
    이름의 범위: 문헌과 같게 맞춘 것은 데이터 프로토콜뿐이고 모델·학습 레시피는 우리 것이다. '문헌 설정'이라고
    부르지 않는다. 표·대시보드·논문에는 '모델·학습 레시피는 우리 것'과 아래 8축을 함께 적는다.
    'vggdvslit baseline X vs 문헌 Y' 같은 비교 문장은 쓰지 않는다. 비교 기준 수치는 UniSparse dense 83.50 +-0.48
    하나만, 레시피가 다르다는 단서를 붙여 옆에 둔다.
    configs_paper/vggdvs/config_sweep.py 를 복사해 네 값만 바꿨다. exp_set_name·time_step 은 그 줄을 고쳤고,
    batch_size·data_aug_mix 는 원래 줄 바로 뒤에 덮어쓰는 줄을 넣었다. 나머지는 한 글자도 같다:
      exp_set_name 'EIP-SNN-26_vggdvslit-src', time_step 4 -> 10, batch_size 32 -> 40,
      data_aug_mix 'nda' -> 'None' (이 조합의 로더 datasets/cifar10_dvs_lit.py 는 이 값을 안 읽는다.
      'nda' 로 두면 nda 증강을 한 것처럼 보이므로 끈다).
    문헌(UniSparse 2026 기준, Shi 2024 는 같은 코드이되 보간 기본값 bilinear, Activity Pruning 은 DVS 설정 미확인)과 같게 맞춘
    데이터 프로토콜은 로더와 run_paper.py 가 한다: 이벤트 수 10등분 프레임 (개수 그대로, 눈금 1.0),
    48x48 bicubic+antialias, 클립 단위 증강 (flip + roll/rotate/cutout 중 하나), 클래스별 900/100 분할
    (시험 1000장 전부 = batch 40 의 배수), T·B 묶음 BN (COMBO_FLAGS bn_per_step).
    손실·평가: lib_snn/tet_loss.py out='step' -- 누적 막전위 출력에서 스텝 출력 u_t 를 복원해 문헌식 TET
    (스텝별 CE 평균, MSE 항 0) 과 문헌 평가 (argmax sum_t u_t) 를 쓴다.
    다른 8축 (우리 것 / UniSparse):
      (1) 뉴런 LIF 0.9 + reset_by_sub + asym 0.6 / tau=2 decay_input + hard reset 0 + ATan + detach_reset.
          neurons.py 는 학습 경로이고 ours_inter_ch 의 vmem·final_step 동작이 이 뉴런에 맞춰져 있다.
      (2) 출력층 FC10 + n_fc 누적 막전위 / 스텝별 Linear 4608->100 + AvgPool1d(10). 모델은 그대로 둔다.
      (3) AdamW 6e-3, wd 2e-2, COS (워밍업 10에폭 lr_init 1e-5 -> 6e-3, 하한 1e-5), 200에폭, 배치 40
          / SGD nesterov 0.025, wd 0, 워밍업 0, cos 하한 0, 300에폭, 배치 64.
          이 프레임워크에서 검증된 레시피다. 문헌 레시피는 여기서 미검증.
      (4) label smoothing 0.1 / 0. lambda = rho*L_task/R 이라 L_task 가 0 으로 가면 규제가 꺼진다.
      (5) 초기화 glorot_uniform / kaiming_normal fan_out.
      (6) BN momentum·eps 0.99·1e-4 / 0.9·1e-5 (lib_snn 층 정의).
      (7) fp32 / AMP.
      (8) conv bias 있음 (lib_snn.layers.Conv2D 기본, BN 바로 앞) / 없음 (bias=False).
    검증셋 = 시험셋 (같은 1000장): val_acc 최고 에폭 = best epoch on test (문헌 관례). 기존 T=4 조합의 val
    (tfds 앞 10% 992장) 과 성격이 달라 별도 표에 두고 같은 표·그래프에 섞지 않는다.
    스파이크: s_count 는 n_in 아날로그 입력 합을 포함하고 T=10 합이다. 문헌 정의(발화율·SOP)에 맞춘 값은
    lib_snn/lit_spike_metrics.py 가 n_in 을 빼고 따로 센다 (train.log [LITSPK], lit_spikes.csv).
    s_count 와 뉴런별 npz 총합의 11% 불일치가 풀릴 때까지는 같은 조합 안의 상대 감소율만 보고한다.
    아래는 vggdvs 원본의 머리 주석 그대로다.

    run_paper.py 의 vggdvs 조합 원본 (VGGSNN x CIFAR10-DVS, 26-09-29).
    vggc10 원본(_sweep_wta_rev/lambda_1e-07/config_sweep.py)을 복사해 모델·데이터셋·학습
    하이퍼파라미터만 바꿨다. 규제 블록은 vggc10 원본과 한 글자도 다르지 않다 -- 방법별
    플래그는 run_paper.py 가 config.set() 앞에 덮어쓴다.
    학습 레시피는 Surro(config_snn_training_surro.py 의 VSNN_DVS)를 따른다:
      T=4, 48x48, batch 32, 200 epoch, AdamW + COS (1e-5 -> 6e-3), wd 2e-2, label smoothing 0.1,
      avg pooling, 워밍업 10에폭 (lib_snn/model_builder.py 의 CIFAR10_DVS 분기).
      증강은 Surro 의 'nda' (26-09-30 이식, datasets/cifar10_dvs.py): resize + 좌우 flip 뒤
      프레임마다 roll/rotate/shear/cutout 중 하나. 09-29~30 에 돈 vggdvs base 시드 1-4 는
      nda 이식 전 기본 증강(pad 3 + random crop + 좌우/상하 flip, 워밍업 20에폭)이다 -- 섞지 않는다.
      surrogate 는 asym(0.6) 유지 (Surro 원래 레시피와 다르다, 사용자 결정).
    _*/ 는 .gitignore 대상이라 기존 원본들은 sejong 에 pull 로 안 간다. 이 파일은
    configs_paper/ 아래에 두어 git 으로 두 서버가 같은 원본을 받는다.
'''

# GPU setting
import os
os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

#
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  # 0: show all, 1: hide info, 2: hide info&warning, 3: hide all (info, warning, error)

#
from config import config
conf = config.flags

#
#conf.debug_mode = True
#conf.verbose_snn_train = True
#conf.debug_grad = True


#
#conf.exp_set_name='EIP-SNN_detail'
#conf.exp_set_name='EIP-SNN_sc_loss_schedule'
#conf.exp_set_name='integer_spike_test'
#conf.exp_set_name='ASY-SNN'
conf.exp_set_name='EIP-SNN-26_vggdvslitlk5-src'


conf.root_model_save=conf.exp_set_name

#
#conf.mode='inference'
##conf.batch_size_inf=100
#conf.batch_size=400
#conf.batch_size=200
#conf.batch_size=180
#conf.batch_size=120
#conf.time_step=2
#conf.name_model_load='./models/VGG16_AP_CIFAR100/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-z'

# imagenet config - ResNet18 (2 GPU)
#conf.batch_size=200
#conf.train_epoch = 90
#conf.step_decay_epoch = 30

# cifar10-dvs config
#conf.learning_rate = 0.1
#conf.lmb = 1E-3

#
#conf.learning_rate = 0.04
#conf.lmb = 1E-3
#conf.time_step = 10
#conf.optimizer = 'ADAM'
#conf.lr_schedule = None

#
#conf.train_epoch = 10
#conf.train_epoch = 10
#conf.num_train_data = 1000

#conf.model='VGG11'
#conf.model='VGG16'
#conf.model='ResNet18'
conf.model='VGGSNN'
#conf.model='ResNet20'
#conf.model='ResNet32'
#conf.model='ResNet20_SEW'   # spike-element-wise block
#conf.model = 'Spikformer'
#conf.model = 'Spikformer_tb'

#conf.dataset='CIFAR10'
#conf.dataset='ImageNet'
conf.dataset='CIFAR10_DVS'

conf.pooling_vgg = 'avg'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'


#conf.use_bn=False

#conf.n_init_vth = 0.3

conf.leak_const_init = 0.9
# vggdvslitlk5 (26-11-03): 누설 0.9 -> 0.5 (문헌 tau=2 의 충전식). 리셋은 reset_by_sub 그대로
conf.leak_const_init = 0.5
#conf.leak_const_train = True




conf.optimizer = 'ADAMW'
conf.lr_schedule = 'COS'

conf.nn_mode = 'SNN'
#conf.nn_mode = 'ANN'

conf.n_init_vth = 1.0

# VGGSNN-DVS: Surro 레시피 200 epoch. T=4 는 flags 기본값과 같지만 로더가 T 로 프레임을
# 자르므로(datasets/cifar10_dvs.py: num_frames = conf.time_step) 명시한다.
# 입력 48x48 은 flags 기본값 cifar10_dvs_img_size=48 / crop 54 그대로다.
conf.train_epoch = 200
# vggdvslit: 문헌 T=10 (UniSparse configs/cifar10dvs_vggsnn.yaml). 캐시 키 T10_s48 과 로더가 대조한다.
conf.time_step = 10

#
if conf.model=='VGGSNN':
    if conf.dataset=='CIFAR10_DVS':
        # VGGSNN-DVS (Surro: config_snn_training_surro.py 의 CIFAR10_DVS 분기)
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    else:
        assert False
elif conf.model=='VGG16':
    if conf.dataset=='CIFAR10':
        # VGG-C10
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    elif conf.dataset=='CIFAR100':
        # VGG-C100
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    else:
        assert False
elif conf.model=='ResNet19':
    if conf.dataset=='CIFAR10':
        # R19-C10
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    elif conf.dataset=='CIFAR100':
        # R19-C100
        conf.learning_rate_init = 1E-4
        conf.learning_rate = 5E-3
        conf.weight_decay_AdamW = 4E-2
    else:
        assert False
elif conf.model=='Spikformer':
    # spikformer
    conf.learning_rate_init = 1E-4
    conf.learning_rate = 5E-3
    conf.weight_decay_AdamW = 2E-2
else:
    assert False

# Surro 레시피 batch 32. 9000 % 32 = 8 이라 로더가 drop_remainder 로 버린다 (val 은 992장).
conf.batch_size = 32
# vggdvslit: 40 이면 9000/40=225, 1000/40=25 로 학습·시험 모두 버리는 표본이 없다 (모델 배치 고정).
conf.batch_size = 40
conf.label_smoothing=0.1
conf.debug_lr = True
conf.lmb=1E-3
conf.regularizer=None
#conf.data_aug_mix='mixup'

conf.mix_off_iter = 500*200
conf.mix_alpha = 0.5

conf.randaug_en = True
# DVS 로더는 randaug / rand_erase / cutmix 를 읽지 않는다 (image_cls 경로 전용). Surro 도 같다.
# 켜 둔 채면 체크포인트 경로명에 _ra _re _cm 이 붙어 안 한 증강을 한 것처럼 보이므로 끈다.
# 학습에는 영향이 없다.
# data_aug_mix='nda' 는 DVS 로더가 읽는 유일한 값이다 (Surro config_snn_training_surro.py 399행).
conf.randaug_en = False
conf.data_aug_mix = 'nda'
# vggdvslit: 로더(datasets/cifar10_dvs_lit.py)가 클립 단위 문헌 증강을 직접 한다. 이 값은 안 읽는다.
conf.data_aug_mix = 'None'
conf.randaug_mag = 0.9
conf.randaug_mag_std = 0.4
conf.randaug_n = 1
conf.randaug_rate = 0.5

conf.rand_erase_en = False

# integer spike - test
#conf.binary_spike = False
#conf.integer_spike = True
#conf.time_step=1


#
conf.exp_set_name = conf.exp_set_name+'_asym'
conf.fire_surro_grad_func = 'asym'

if conf.fire_surro_grad_func=='asym':
    if conf.model in ('VGG16', 'VGGSNN'):
        # VGGSNN 도 VGG 계열 값 (Surro asy_height_val: VGG16 0.6)
        conf.surrogate_bias = 0.6
    elif 'ResNet' in conf.model:
        conf.surrogate_bias = 0.8
    elif conf.model=='Spikformer':
        conf.surrogate_bias = 0.6
    else:
        assert False



#
#if False:
if True:
    if True:    # proposed method
    #if False:
        conf.reg_spike_out=True
        conf.reg_spike_out_const=1e-07
        conf.reg_spike_out_alpha=7  # temperature
        #conf.reg_spike_rate_alpha=8E-1  # coefficient of reg. rate
        conf.reg_spike_out_sc=True
        #conf.reg_spike_out_sc_wta=False
        #conf.reg_spike_out_sc_train=True
        conf.reg_spike_out_sc_sm=True
        #conf.reg_spike_out_sc_sq=True
        conf.reg_spike_out_norm=True
        #conf.reg_spike_out_norm_sq=True
        conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)

        conf.reg_spike_log_detail=True   # per-layer regularization metrics logging

        #
        #conf.reg_spike_out_sc_sm_wo_tmp=True
        #conf.reg_spike_out_sc_sm_wo_spa=True


        #
        conf.sc_loss_scd = False
        #conf.sc_loss_scd = True
        conf.sc_loss_scd_st_ep = 200
        conf.sc_loss_scd_end_ep = 300
        #conf.sc_loss_layer_wise = False

    else:   # previous work
        conf.reg_spike_out = True
        conf.reg_spike_out_const = 5E-9
        conf.reg_spike_out_alpha = 4  # temperature
        # conf.reg_spike_rate_alpha=8E-1  # coefficient of reg. rate
        #conf.reg_spike_out_sc = True
        #conf.reg_spike_out_sc_wta=False
        # conf.reg_spike_out_sc_train=True
        #conf.reg_spike_out_sc_sm = True
        # conf.reg_spike_out_sc_sq=True
        conf.reg_spike_out_norm = True
        #conf.reg_spike_out_norm_sq=True

#
config.set()
