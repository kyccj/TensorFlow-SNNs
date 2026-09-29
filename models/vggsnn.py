"""VGGSNN (CIFAR10-DVS 용 8-conv VGG) — Surro/models/vggsnn.py 에서 옮김 (26-09-29).

구조 (TET, Deng et al. ICLR 2022 의 VGGSNN 과 같은 채널 배치):
    64, 128, AP, 256, 256, AP, 512, 512, AP, 512, 512, AP, FC(classes)
입력은 [T, 48, 48, 2] 이벤트 프레임 (datasets/cifar10_dvs.py). 48 -> 3x3 이 되어 FC 입력은 4608.

Surro 원본과 다른 점 (이 저장소 VGG16(models/vgg16_func.py) 관례에 맞춤):
  - conv1 에 use_bn 을 넘기지 않는다. Surro 는 Conv2D(use_bn=True) 와 별도 bn_conv1 을 둘 다
    달아 conv1 뒤에 BN 이 두 번 걸린다 (layers.py:614 의 내부 BN + bn_conv1).
    vgg16_func.py 는 같은 이유로 그 인자를 주석 처리했다.
  - BN·Input 에 dtype=tf.float32 를 명시한다 (vgg16_func.py 와 같게. mixed precision 대비).
층 이름(n_conv1 ... n_conv4_1, n_fc)은 Surro 그대로 둔다. 규제 경로(neurons.py)는 이름이 아니라
loc=='HID' 로 층을 고르므로 n_in(IN)·n_fc(OUT) 는 빠지고 8개 conv 뉴런층만 규제된다.
"""
import tensorflow as tf

import lib_snn


def VGGSNN(
        batch_size,
        input_shape,
        conf,
        model_name,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        dataset_name=None,
        **kwargs):

    data_format = conf.data_format

    if conf.nn_mode == 'ANN':
        dropout_conv_r = [0.2, 0.2, 0.0]      # DNN training
    elif conf.nn_mode == 'SNN':
        dropout_conv_r = [0.0, 0.0, 0.0]      # SNN training (Surro 와 같음)
    else:
        assert False

    initial_channels = kwargs.pop('initial_channels', None)
    if initial_channels is None:
        initial_channels = 64

    use_bn_feat = conf.use_bn

    channels = initial_channels

    k_init = 'glorot_uniform'

    # pooling -- VGGSNN 표준은 average pooling (config 에서 pooling_vgg='avg')
    if conf.pooling_vgg == 'max':
        pool = lib_snn.layers.MaxPool2D
    elif conf.pooling_vgg == 'avg':
        pool = lib_snn.layers.AveragePooling2D
    else:
        assert False

    if conf.nn_mode == 'ANN':
        act_type = 'relu'
        act_type_out = 'softmax'
    else:
        act_type = conf.n_type
        act_type_out = conf.n_type

    # 첫 층은 입력이 POISSON 일 때만 tdBN (Surro 와 같음. DVS 는 REAL 입력이라 끔)
    tdbn_first_layer = conf.nn_mode == 'SNN' and conf.input_spike_mode == 'POISSON' and conf.tdbn
    tdbn = conf.nn_mode == 'SNN' and conf.tdbn

    def conv_bn_act(x, ch, name, en_tdbn):
        syn = lib_snn.layers.Conv2D(ch, 3, padding='SAME', kernel_initializer=k_init, name=name)(x)
        if use_bn_feat:
            norm = lib_snn.layers.BatchNormalization(en_tdbn=en_tdbn, dtype=tf.float32,
                                                     name='bn_' + name)(syn)
        else:
            norm = syn
        return lib_snn.activations.Activation(act_type=act_type, name='n_' + name)(norm)

    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
    x = lib_snn.layers.InputGenLayer(name='in')(img_input)
    if conf.nn_mode == 'SNN':
        x = lib_snn.activations.Activation(act_type=act_type, loc='IN', name='n_in')(x)

    # [2, 48, 48] -> [64, 48, 48]
    x = conv_bn_act(x, channels, 'conv1', tdbn_first_layer)
    # [64, 48, 48] -> [128, 48, 48] -> AP -> [128, 24, 24]
    channels = channels * 2
    x = conv_bn_act(x, channels, 'conv1_1', tdbn)
    x = pool((2, 2), (2, 2), name='conv1_1_p')(x)

    # [128, 24, 24] -> [256, 24, 24] -> [256, 24, 24] -> AP -> [256, 12, 12]
    channels = channels * 2
    x = conv_bn_act(x, channels, 'conv2', tdbn)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv2_do')(x)
    x = conv_bn_act(x, channels, 'conv2_1', tdbn)
    x = pool((2, 2), (2, 2), name='conv2_1p')(x)

    # [256, 12, 12] -> [512, 12, 12] -> [512, 12, 12] -> AP -> [512, 6, 6]
    channels = channels * 2
    x = conv_bn_act(x, channels, 'conv3', tdbn)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_do')(x)
    x = conv_bn_act(x, channels, 'conv3_1', tdbn)
    x = pool((2, 2), (2, 2), name='conv3_1p')(x)

    # [512, 6, 6] -> [512, 6, 6] -> [512, 6, 6] -> AP -> [512, 3, 3]
    x = conv_bn_act(x, channels, 'conv4', tdbn)
    x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_do')(x)
    x = conv_bn_act(x, channels, 'conv4_1', tdbn)
    x = pool((2, 2), (2, 2), name='conv4_1p')(x)

    # [512, 3, 3] -> [4608] -> [classes]
    x = tf.keras.layers.Flatten(data_format=data_format, name='flatten')(x)
    x = tf.keras.layers.Dropout(dropout_conv_r[2], name='flatten_do')(x)
    syn_fc = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init, name='fc')(x)
    a_p = lib_snn.activations.Activation(act_type=act_type_out, loc='OUT', name='n_fc')(syn_fc)
    if conf.nn_mode == 'SNN':
        a_p = lib_snn.activations.Activation(act_type='softmax', name='a_predictions')(a_p)

    model = lib_snn.model.Model(img_input, a_p, batch_size, input_shape, classes, conf, name=model_name)

    return model
