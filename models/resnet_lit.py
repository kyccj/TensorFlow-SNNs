"""문헌 비교용 ResNet 변형 (26-11-10, 11-10 박사님 미팅 지시: 문헌과 같은 구조·같은 T 에서 baseline·ours 를 돌려
그 논문이 보고한 수치를 옆에 놓는다).

models/resnet.py 의 기존 정의는 건드리지 않는다. 여기 두 모델은 새 이름으로만 불린다 (models/models.py 의
model_sel_sc 에 추가). 층 이름은 기존 ResNet 과 같은 관례 ('in', 'n_in', 'conv1_conv', 'avg_pool', 'predictions',
'predictions_n', 'predictions_a') 를 따른다 -- lib_snn 이 'predictions' 이름을 직접 찾는 곳이 있다 (layers.py).

ResNet20_W32  -- Activity Pruning for Efficient Spiking Neural Networks (NeurIPS 2025) 의 CIFAR-10 ResNet-20.
    공개 코드 putshua/Activity-Pruning-SNN models/resnet_cifar.py (커밋 01c4810): 폭 16*2 = 32/64/128, 단계당 기본 블록 3개,
    줄기 Conv3x3(32)-BN-LIF, 블록 = conv-bn-lif-conv-bn, (+shortcut), lif. shortcut 은 stride != 1 이거나 채널이 바뀔 때만
    1x1 conv + BN, 아니면 항등. 머리 = 전역 평균 풀링 + Linear(128->10).
    기존 ResNet20 (폭 64/128/256, 단계마다 첫 블록에 1x1 conv shortcut) 과 다른 점은 폭과 첫 단계 shortcut 둘이다.
    블록은 models.resnet.block_basic 을 그대로 쓴다 (conv_shortcut 만 위 규칙으로 정한다).

ResNet18_PreAct -- Synaptic Interaction Penalty (Spiking Synaptic Penalty, arXiv 2302.01500, TMLR 2024) 의 ResNet18.
    그 논문 Table 5: C(64)k3s1p1 - {ResA(64) ResA(64)} {ResB(128) ResA(128)} {ResB(256) ResA(256)} {ResB(512) ResA(512)}
    - BN - S - C(10)k1 - GAP. ResA/ResB 는 사전 활성화(BN-S-Conv) 블록이고 잔차 흐름은 아날로그 합이다.
    shortcut 은 1x1 conv (stride 2) 로 둔다: Table 5 본문은 ResB 의 둘째 경로를 C(x)k(3)s(1)p(1) 로 적었지만,
    같은 논문 Table 1 의 ResNet18 E_SNN/E_AC = 553,730,048 (모든 뉴런 발화 가정) 은 1x1 stride-2 shortcut 일 때만
    정확히 나온다 (3x3 이면 604,061,696). 그림 파일명도 preresnet18 이라 표준 PreActResNet18 (shortcut 이 첫 BN-S
    출력을 가지와 같이 씀) 과 같은 연결로 둔다.
    머리 C(10)k1 - GAP 는 GAP - Dense(10) 과 함수로 같다 (둘 다 선형, 편향도 같은 자리). 그래서 기존 관례대로
    GAP - Dense('predictions') - 출력 뉴런 - softmax 로 둔다.
    줄기 conv 뒤에는 BN·뉴런이 없다 (Table 5 그대로). 첫 뉴런은 첫 블록의 BN-S 다.

두 모델 모두 우리 레시피(뉴런 LIF·누설·Vth·서로게이트·학습 설정)는 원본 config 를 그대로 따른다. 구조만 문헌에 맞춘다.
"""
import tensorflow as tf

import lib_snn
from models.resnet import block_basic, tdbn, act_type, act_type_out


def _stem_input(batch_size, input_shape, conf):
    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = lib_snn.layers.InputGenLayer(name='in')(img_input)
    if conf.nn_mode == 'SNN':
        x = lib_snn.activations.Activation(act_type=act_type, loc='IN', name='n_in')(x)
    return img_input, x


def _head(x, classes, conf):
    # models.resnet.ResNet 의 include_top 머리와 같다 (add_dense_layer 없음)
    x = lib_snn.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = lib_snn.layers.Dense(classes, last_layer=True, name='predictions')(x)
    x = lib_snn.activations.Activation(act_type=act_type_out, loc='OUT', name='predictions_n')(x)
    if conf.nn_mode == 'SNN':
        x = lib_snn.activations.Activation(act_type='softmax', name='predictions_a')(x)
    return x


def ResNet20_W32(input_shape, conf, include_top, weights, classes, **kwargs):
    assert include_top, 'ResNet20_W32: include_top=False 미지원'
    batch_size = kwargs.pop('batch_size')
    model_name = kwargs.pop('model_name')
    img_input, x = _stem_input(batch_size, input_shape, conf)

    x = lib_snn.layers.Conv2D(32, 3, strides=1, padding='same', name='conv1_conv')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='conv1_conv_bn')(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='conv1_conv_n')(x)

    in_ch = 32
    for stage, (ch, stride) in enumerate([(32, 1), (64, 2), (128, 2)], start=2):
        for b in range(1, 4):
            s = stride if b == 1 else 1
            # 문헌 코드와 같이 모양이 바뀔 때만 1x1 conv shortcut, 아니면 항등
            x = block_basic(x, ch, stride=s, conv_shortcut=(s != 1 or in_ch != ch),
                            name=f'conv{stage}_block{b}')
            in_ch = ch

    x = _head(x, classes, conf)
    return lib_snn.model.Model(img_input, x, batch_size, input_shape, classes, conf, name=model_name)


def block_preact(x, filters, stride=1, conv_shortcut=False, name=None):
    """사전 활성화 기본 블록 (BN-S-Conv-BN-S-Conv, 뒤에 활성 없음). 입력 x 는 아날로그 잔차 흐름이다.
    conv_shortcut 이면 첫 BN-S 출력에 1x1 conv (stride) 를 건다 (BN 없음), 아니면 x 를 그대로 더한다."""
    pre = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=name + '_conv1_prebn')(x)
    pre = lib_snn.activations.Activation(act_type=act_type, name=name + '_conv1_n')(pre)
    if conv_shortcut:
        shortcut = lib_snn.layers.Conv2D(filters, 1, strides=stride, name=name + '_conv0')(pre)
    else:
        shortcut = lib_snn.layers.Identity(name=name + '_conv0_i')(x)
    y = lib_snn.layers.Conv2D(filters, 3, strides=stride, padding='SAME', name=name + '_conv1')(pre)
    y = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=name + '_conv2_prebn')(y)
    y = lib_snn.activations.Activation(act_type=act_type, name=name + '_conv2_n')(y)
    y = lib_snn.layers.Conv2D(filters, 3, padding='SAME', name=name + '_conv2')(y)
    return lib_snn.layers.Add(name=name + '_out')([shortcut, y])


def ResNet18_PreAct(input_shape, conf, include_top, weights, classes, **kwargs):
    assert include_top, 'ResNet18_PreAct: include_top=False 미지원'
    batch_size = kwargs.pop('batch_size')
    model_name = kwargs.pop('model_name')
    img_input, x = _stem_input(batch_size, input_shape, conf)

    # 줄기: C(64)k3s1p1 뿐 (BN·뉴런 없음, 문헌 Table 5)
    x = lib_snn.layers.Conv2D(64, 3, strides=1, padding='same', name='conv1_conv')(x)

    in_ch = 64
    for stage, (ch, stride) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)], start=2):
        for b in range(1, 3):
            s = stride if b == 1 else 1
            x = block_preact(x, ch, stride=s, conv_shortcut=(s != 1 or in_ch != ch),
                             name=f'conv{stage}_block{b}')
            in_ch = ch

    # 머리 앞 BN-S (문헌 Table 5 의 '- BN - S - C(10)k1 - GAP')
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='post_bn')(x)
    x = lib_snn.activations.Activation(act_type=act_type, name='post_n')(x)
    x = _head(x, classes, conf)
    return lib_snn.model.Model(img_input, x, batch_size, input_shape, classes, conf, name=model_name)
