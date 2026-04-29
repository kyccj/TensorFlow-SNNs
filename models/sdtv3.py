import tensorflow as tf
import lib_snn

from config import config
conf = config.flags

from models.sdtv3_blocks import ms_conv_block, ms_block

# Variant configurations: embed_dims per stage, block depths per stage
_VARIANTS = {
    'tiny':   {'dims': [24,  48,  96,  128], 'depths': [1, 1, 1, 1]},
    'small':  {'dims': [32,  64,  128, 192], 'depths': [1, 1, 2, 2]},
    'medium': {'dims': [48,  96,  192, 240], 'depths': [1, 1, 2, 4]},
}


def sdtv3(batch_size, input_shape, conf, model_name,
          variant='small', classes=1000, include_top=True,
          dataset_name=None, weights=None, **kwargs):

    cfg = _VARIANTS.get(variant)
    assert cfg is not None, f'Unknown SDT-V3 variant: {variant}. Choose from {list(_VARIANTS)}'
    dims = cfg['dims']
    depths = cfg['depths']

    num_heads = conf.sdtv3_num_heads
    k_init = 'glorot_uniform'
    tdbn = (conf.nn_mode == 'SNN') and conf.tdbn

    act_tp = 'relu' if conf.nn_mode == 'ANN' else conf.n_type

    input_tensor = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = lib_snn.layers.InputGenLayer(name='in')(input_tensor)
    if conf.nn_mode == 'SNN':
        x = lib_snn.activations.Activation(act_type=act_tp, loc='IN', name='n_in')(x)

    # 4-stage hierarchical network
    # Stage 1-2: purely convolutional (MS_ConvBlock)
    # Stage 3-4: mix of convolution and transformer (MS_Block added in later blocks)
    for stage_idx, (dim, depth) in enumerate(zip(dims, depths)):
        # Downsample at the start of each stage via stride-2 conv
        x = lib_snn.layers.Conv2D(dim, kernel_size=3, strides=2, padding='SAME',
                                   kernel_initializer=k_init, use_bias=False,
                                   name=f's{stage_idx+1}_ds_conv')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                               name=f's{stage_idx+1}_ds_bn')(x)
        x = lib_snn.activations.Activation(act_type=act_tp,
                                            name=f's{stage_idx+1}_ds_n')(x)

        for blk_idx in range(depth):
            # Stages 0-1: all convolutional blocks
            # Stages 2-3: first half convolutional, second half transformer
            use_attn = (stage_idx >= 2) and (blk_idx >= depth // 2)
            if use_attn:
                x = ms_block(x, dim, num_heads=num_heads, mlp_ratio=4,
                              name_prefix=f's{stage_idx+1}_tblk{blk_idx}',
                              k_init=k_init, tdbn=tdbn)
            else:
                x = ms_conv_block(x, dim, mlp_ratio=2,
                                  name_prefix=f's{stage_idx+1}_cblk{blk_idx}',
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
