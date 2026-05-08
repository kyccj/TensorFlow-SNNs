import tensorflow as tf
import lib_snn

from config import config
conf = config.flags

from models.sdtv3_blocks import ms_conv_block, ms_block

# 5-stage hierarchical backbone (Spiking_vit_MetaFormer_Spike_SepConv):
#   Stage 1a: DS(7×7, s=2) → dims[0]//2 + 1× MS_ConvBlock  (no spike before DS)
#   Stage 1b: DS(3×3, s=2) → dims[0]    + 1× MS_ConvBlock
#   Stage 2:  DS(3×3, s=2) → dims[1]    + 2× MS_ConvBlock
#   Stage 3:  DS(3×3, s=2) → dims[2]    + 6× MS_Block
#   Stage 4:  DS(3×3, s=1) → dims[3]    + 2× MS_Block  (stride=1: channel-only projection)

_VARIANTS = {
    'tiny':   [24,  48,  96,  128],
    'small':  [32,  64,  128, 192],
    'medium': [48,  96,  192, 240],
    'large':  [64,  128, 256, 360],
}


def sdtv3(batch_size, input_shape, conf, model_name,
          variant='small', classes=1000, include_top=True,
          dataset_name=None, weights=None, **kwargs):

    dims = _VARIANTS.get(variant)
    assert dims is not None, \
        f'Unknown SDT-V3 variant: {variant}. Choose from {list(_VARIANTS)}'

    num_heads = conf.sdtv3_num_heads
    k_init    = 'glorot_uniform'
    tdbn      = (conf.nn_mode == 'SNN') and conf.tdbn
    act_tp    = 'relu' if conf.nn_mode == 'ANN' else conf.n_type

    input_tensor = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = lib_snn.layers.InputGenLayer(name='in')(input_tensor)
    if conf.nn_mode == 'SNN':
        x = lib_snn.activations.Activation(act_type=act_tp, loc='IN', name='n_in')(x)

    # ---- Stage 1a: 7×7 DS, no spike before conv (first_layer=True) --------
    d1a = dims[0] // 2
    x = lib_snn.layers.Conv2D(d1a, kernel_size=7, strides=2, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name='s1a_ds_conv')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name='s1a_ds_bn')(x)
    x = ms_conv_block(x, d1a, mlp_ratio=4, name_prefix='s1a_cblk0',
                      k_init=k_init, tdbn=tdbn)

    # ---- Stage 1b: 3×3 DS, spike before conv (first_layer=False) ----------
    x = lib_snn.activations.Activation(act_type=act_tp, name='s1b_ds_n')(x)
    x = lib_snn.layers.Conv2D(dims[0], kernel_size=3, strides=2, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name='s1b_ds_conv')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name='s1b_ds_bn')(x)
    x = ms_conv_block(x, dims[0], mlp_ratio=4, name_prefix='s1b_cblk0',
                      k_init=k_init, tdbn=tdbn)

    # ---- Stage 2: 3×3 DS + 2× MS_ConvBlock --------------------------------
    x = lib_snn.activations.Activation(act_type=act_tp, name='s2_ds_n')(x)
    x = lib_snn.layers.Conv2D(dims[1], kernel_size=3, strides=2, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name='s2_ds_conv')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name='s2_ds_bn')(x)
    for i in range(2):
        x = ms_conv_block(x, dims[1], mlp_ratio=4, name_prefix=f's2_cblk{i}',
                          k_init=k_init, tdbn=tdbn)

    # ---- Stage 3: 3×3 DS + 6× MS_Block ------------------------------------
    x = lib_snn.activations.Activation(act_type=act_tp, name='s3_ds_n')(x)
    x = lib_snn.layers.Conv2D(dims[2], kernel_size=3, strides=2, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name='s3_ds_conv')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name='s3_ds_bn')(x)
    for i in range(6):
        x = ms_block(x, dims[2], num_heads=num_heads, mlp_ratio=4,
                     name_prefix=f's3_tblk{i}', k_init=k_init, tdbn=tdbn)

    # ---- Stage 4: 3×3 DS (stride=1) + 2× MS_Block -------------------------
    x = lib_snn.activations.Activation(act_type=act_tp, name='s4_ds_n')(x)
    x = lib_snn.layers.Conv2D(dims[3], kernel_size=3, strides=1, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name='s4_ds_conv')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name='s4_ds_bn')(x)
    for i in range(2):
        x = ms_block(x, dims[3], num_heads=num_heads, mlp_ratio=4,
                     name_prefix=f's4_tblk{i}', k_init=k_init, tdbn=tdbn)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last',
                                                    name='gap')(x)
        x = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init,
                                  temporal_mean_input=True, dtype=tf.float32,
                                  name='predictions')(x)
        x = lib_snn.activations.Activation(act_type='softmax', loc='OUT',
                                            dtype=tf.float32, name='n_predictions')(x)

    model = lib_snn.model.Model(input_tensor, x, batch_size, input_shape,
                                 classes, conf, name=model_name)
    return model
