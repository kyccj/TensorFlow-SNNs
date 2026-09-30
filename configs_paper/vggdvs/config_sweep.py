'''
    Configuration for SNN direct training

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
conf.exp_set_name='EIP-SNN-26_vggdvs-src'


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
conf.time_step = 4

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
