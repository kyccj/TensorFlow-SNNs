import tensorflow as tf
import lib_snn

from config import config
conf = config.flags

from models.sdtv3_blocks import ms_conv_block, ms_block

# Original block structure (hardcoded in paper, same for all variants):
#   Stage 1: 2 x MS_ConvBlock
#   Stage 2: 2 x MS_ConvBlock
#   Stage 3: 6 x MS_Block  (transformer)
#   Stage 4: 2 x MS_Block  (transformer)
_STAGE_CONV_DEPTHS = [2, 2]   # stages 1-2: ConvBlock counts
_STAGE_ATTN_DEPTHS = [6, 2]   # stages 3-4: Transformer block counts

# Only embed_dims differ between variants
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
    k_init = 'glorot_uniform'
    tdbn = (conf.nn_mode == 'SNN') and conf.tdbn
    act_tp = 'relu' if conf.nn_mode == 'ANN' else conf.n_type

    input_tensor = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = lib_snn.layers.InputGenLayer(name='in')(input_tensor)
    if conf.nn_mode == 'SNN':
        x = lib_snn.activations.Activation(act_type=act_tp, loc='IN', name='n_in')(x)

    # Stages 1-2: ConvBlock only (stride-2 downsample + MS_ConvBlock x n)
    for stage_idx, (dim, n_blk) in enumerate(zip(dims[:2], _STAGE_CONV_DEPTHS)):
        x = lib_snn.layers.Conv2D(dim, kernel_size=3, strides=2, padding='SAME',
                                   kernel_initializer=k_init, use_bias=False,
                                   name=f's{stage_idx+1}_ds_conv')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                               name=f's{stage_idx+1}_ds_bn')(x)
        x = lib_snn.activations.Activation(act_type=act_tp,
                                            name=f's{stage_idx+1}_ds_n')(x)
        for blk_idx in range(n_blk):
            x = ms_conv_block(x, dim, mlp_ratio=2,
                               name_prefix=f's{stage_idx+1}_cblk{blk_idx}',
                               k_init=k_init, tdbn=tdbn)

    # Stages 3-4: Transformer block only (stride-2 downsample + MS_Block x n)
    for stage_idx, (dim, n_blk) in enumerate(zip(dims[2:], _STAGE_ATTN_DEPTHS)):
        real_stage = stage_idx + 3   # display as stage 3, 4
        x = lib_snn.layers.Conv2D(dim, kernel_size=3, strides=2, padding='SAME',
                                   kernel_initializer=k_init, use_bias=False,
                                   name=f's{real_stage}_ds_conv')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                               name=f's{real_stage}_ds_bn')(x)
        x = lib_snn.activations.Activation(act_type=act_tp,
                                            name=f's{real_stage}_ds_n')(x)
        for blk_idx in range(n_blk):
            x = ms_block(x, dim, num_heads=num_heads, mlp_ratio=4,
                          name_prefix=f's{real_stage}_tblk{blk_idx}',
                          k_init=k_init, tdbn=tdbn)

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
