import tensorflow as tf
import lib_snn

from config import config
conf = config.flags


def _act_type():
    return 'relu' if conf.nn_mode == 'ANN' else conf.n_type


# ---------------------------------------------------------------------------
# RepConv: 1x1 pointwise conv + 3x3 depthwise conv, both with BN, summed
# At inference, BN can be folded into each conv (fold_bn=True).
# ---------------------------------------------------------------------------
def repconv(x, channels, name_prefix='repconv', k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()

    pw = lib_snn.layers.Conv2D(channels, kernel_size=1, padding='SAME',
                                kernel_initializer=k_init, use_bias=False,
                                name=name_prefix + '_pw')(x)
    pw_bn = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                               name=name_prefix + '_pw_bn')(pw)

    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='SAME',
                                          use_bias=False,
                                          name=name_prefix + '_dw')(x)
    dw_bn = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                               name=name_prefix + '_dw_bn')(dw)

    out = lib_snn.layers.Add(name=name_prefix + '_add')([pw_bn, dw_bn])
    return out


# ---------------------------------------------------------------------------
# SepConv: MobileNetV2-style inverted separable conv (expansion=2)
# BN -> Spike -> pw_expand -> BN -> Spike -> dw3x3 -> BN -> Spike -> pw_project -> BN
# ---------------------------------------------------------------------------
def sepconv(x, in_channels, out_channels, expansion=2,
            name_prefix='sepconv', k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()
    mid_channels = in_channels * expansion

    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn0')(x)
    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n0')(x)

    pw_exp = lib_snn.layers.Conv2D(mid_channels, kernel_size=1, padding='SAME',
                                    kernel_initializer=k_init, use_bias=False,
                                    name=name_prefix + '_pw_exp')(x)
    pw_exp_bn = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                                   name=name_prefix + '_pw_exp_bn')(pw_exp)
    pw_exp_n = lib_snn.activations.Activation(act_type=act_tp,
                                               name=name_prefix + '_n1')(pw_exp_bn)

    dw = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='SAME',
                                          use_bias=False,
                                          name=name_prefix + '_dw')(pw_exp_n)
    dw_bn = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                               name=name_prefix + '_dw_bn')(dw)
    dw_n = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n2')(dw_bn)

    pw_proj = lib_snn.layers.Conv2D(out_channels, kernel_size=1, padding='SAME',
                                     kernel_initializer=k_init, use_bias=False,
                                     name=name_prefix + '_pw_proj')(dw_n)
    pw_proj_bn = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                                    name=name_prefix + '_pw_proj_bn')(pw_proj)
    return pw_proj_bn


# ---------------------------------------------------------------------------
# MS_MLP: two Conv1D(1x1) layers for token-wise MLP
# Input/output shape: [B, N, C]
# Conv1D(kernel=1) on channels_last is equivalent to a shared Dense over N.
# ---------------------------------------------------------------------------
def ms_mlp(x, in_features, hidden_features, out_features,
           name_prefix='ms_mlp', k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()

    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn1')(x)
    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n1')(x)
    x = tf.keras.layers.Conv1D(hidden_features, kernel_size=1, padding='valid',
                                use_bias=False, data_format='channels_last',
                                kernel_initializer=k_init,
                                name=name_prefix + '_fc1')(x)

    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn2')(x)
    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n2')(x)
    x = tf.keras.layers.Conv1D(out_features, kernel_size=1, padding='valid',
                                use_bias=False, data_format='channels_last',
                                kernel_initializer=k_init,
                                name=name_prefix + '_fc2')(x)
    return x


# ---------------------------------------------------------------------------
# MS_Attention: spike-driven multi-head attention with RepConv Q/K/V
# Input shape: [B, H, W, C]  (spatial, channels_last)
# Attention: (Q @ K^T) @ V * scale  (no softmax)
# ---------------------------------------------------------------------------
def ms_attention(x, dim, num_heads=8, attn_scale=None,
                 name_prefix='ms_attn', k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()
    head_dim = dim // num_heads
    scale = attn_scale if attn_scale is not None else (head_dim ** -0.5)

    B = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    N = H * W if (H is not None and W is not None) else None

    # Q, K, V via RepConv + BN + Spike
    q = repconv(x, dim, name_prefix=name_prefix + '_q', k_init=k_init, tdbn=tdbn)
    q = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_q_bn')(q)
    q = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_q_n')(q)

    k = repconv(x, dim, name_prefix=name_prefix + '_k', k_init=k_init, tdbn=tdbn)
    k = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_k_bn')(k)
    k = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_k_n')(k)

    v = repconv(x, dim, name_prefix=name_prefix + '_v', k_init=k_init, tdbn=tdbn)
    v = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_v_bn')(v)
    v = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_v_n')(v)

    # Flatten spatial: [B, H, W, C] -> [B, N, num_heads, head_dim]
    q = tf.keras.layers.Reshape((N, num_heads, head_dim),
                                 name=name_prefix + '_q_reshape')(q)
    k = tf.keras.layers.Reshape((N, num_heads, head_dim),
                                 name=name_prefix + '_k_reshape')(k)
    v = tf.keras.layers.Reshape((N, num_heads, head_dim),
                                 name=name_prefix + '_v_reshape')(v)

    # [B, N, heads, head_dim] -> [B, heads, N, head_dim]
    q = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                name=name_prefix + '_q_perm')(q)
    k = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                name=name_prefix + '_k_perm')(k)
    v = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                name=name_prefix + '_v_perm')(v)

    # Attention: (Q @ K^T) @ V * scale, no softmax
    kt = lib_snn.layers.Permute((1, 3, 2), temporal_batch=True,
                                 name=name_prefix + '_kt_perm')(k)
    attn = lib_snn.layers.Lambda(lambda ts: tf.matmul(ts[0], ts[1]),
                                  name=name_prefix + '_qkt')([q, kt])
    attn = lib_snn.layers.Lambda(lambda ts: tf.matmul(ts[0], ts[1]),
                                  name=name_prefix + '_attnv')([attn, v])
    attn = lib_snn.layers.Lambda(lambda t: t * scale, temporal_batch=True,
                                  name=name_prefix + '_scale')(attn)

    # [B, heads, N, head_dim] -> [B, N, heads, head_dim] -> [B, N, C]
    attn = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                   name=name_prefix + '_attn_perm')(attn)
    attn = tf.keras.layers.Reshape((N, dim), name=name_prefix + '_attn_reshape')(attn)

    # Output projection: Conv2D(1x1) + BN + Spike (back to spatial form)
    attn_spatial = tf.keras.layers.Reshape((H, W, dim),
                                            name=name_prefix + '_proj_reshape')(attn)
    proj = lib_snn.layers.Conv2D(dim, kernel_size=1, padding='SAME',
                                  kernel_initializer=k_init, use_bias=False,
                                  name=name_prefix + '_proj')(attn_spatial)
    proj = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                              name=name_prefix + '_proj_bn')(proj)
    proj = lib_snn.activations.Activation(act_type=act_tp,
                                           name=name_prefix + '_proj_n')(proj)
    return proj


# ---------------------------------------------------------------------------
# MS_ConvBlock: SepConv + MS_MLP with residual connections
# Input/output shape: [B, H, W, C]
# ---------------------------------------------------------------------------
def ms_conv_block(x, dim, mlp_ratio=2,
                  name_prefix='ms_cblk', k_init='glorot_uniform', tdbn=False):
    H = x.shape[1]
    W = x.shape[2]
    N = H * W if (H is not None and W is not None) else None

    # SepConv branch + residual
    residual = x
    sep_out = sepconv(x, dim, dim, expansion=mlp_ratio,
                      name_prefix=name_prefix + '_sep', k_init=k_init, tdbn=tdbn)
    x = lib_snn.layers.Add(name=name_prefix + '_sep_add')([residual, sep_out])

    # MLP branch + residual (flatten spatial for Conv1D, then reshape back)
    residual = x
    x_flat = tf.keras.layers.Reshape((N, dim), name=name_prefix + '_flat')(x)
    mlp_out = ms_mlp(x_flat, dim, int(dim * mlp_ratio), dim,
                     name_prefix=name_prefix + '_mlp', k_init=k_init, tdbn=tdbn)
    x_flat = lib_snn.layers.Add(name=name_prefix + '_mlp_add')(
        [tf.keras.layers.Reshape((N, dim), name=name_prefix + '_res_flat')(residual), mlp_out])
    x = tf.keras.layers.Reshape((H, W, dim), name=name_prefix + '_unflat')(x_flat)
    return x


# ---------------------------------------------------------------------------
# MS_Block: RepConv Attention + MS_MLP with residual connections
# Input/output shape: [B, H, W, C]
# ---------------------------------------------------------------------------
def ms_block(x, dim, num_heads=8, mlp_ratio=4,
             name_prefix='ms_blk', k_init='glorot_uniform', tdbn=False):
    H = x.shape[1]
    W = x.shape[2]
    N = H * W if (H is not None and W is not None) else None

    # Attention branch + residual
    residual = x
    attn_out = ms_attention(x, dim, num_heads=num_heads,
                             name_prefix=name_prefix + '_attn', k_init=k_init, tdbn=tdbn)
    x = lib_snn.layers.Add(name=name_prefix + '_attn_add')([residual, attn_out])

    # MLP branch + residual
    residual = x
    x_flat = tf.keras.layers.Reshape((N, dim), name=name_prefix + '_flat')(x)
    mlp_out = ms_mlp(x_flat, dim, int(dim * mlp_ratio), dim,
                     name_prefix=name_prefix + '_mlp', k_init=k_init, tdbn=tdbn)
    x_flat = lib_snn.layers.Add(name=name_prefix + '_mlp_add')(
        [tf.keras.layers.Reshape((N, dim), name=name_prefix + '_res_flat')(residual), mlp_out])
    x = tf.keras.layers.Reshape((H, W, dim), name=name_prefix + '_unflat')(x_flat)
    return x
