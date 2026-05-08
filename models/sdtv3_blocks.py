import tensorflow as tf
import lib_snn

from config import config
conf = config.flags


def _act_type():
    return 'relu' if conf.nn_mode == 'ANN' else conf.n_type


# ---------------------------------------------------------------------------
# SepConv_Spike: spike→pw+BN→spike→dw+BN→spike→pw+BN
# Conv stages (1-2) use kernel_size=7; transformer stages (3-4) use kernel_size=3
# Input/output shape: [B, H, W, C]
# ---------------------------------------------------------------------------
def sepconv_spike(x, dim, expansion=2, kernel_size=7,
                  name_prefix='sepconv', k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()
    mid = int(dim * expansion)

    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n1')(x)
    x = lib_snn.layers.Conv2D(mid, kernel_size=1, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name=name_prefix + '_pw1')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn1')(x)
    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n2')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='SAME',
                                         use_bias=False,
                                         name=name_prefix + '_dw')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn2')(x)
    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n3')(x)
    x = lib_snn.layers.Conv2D(dim, kernel_size=1, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name=name_prefix + '_pw2')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn3')(x)
    return x


# ---------------------------------------------------------------------------
# MS_ConvBlock: two residual branches
#   1) SepConv_Spike(7×7) + residual
#   2) spike→Conv2D(4×,3×3)→BN→spike→Conv2D(1×,3×3)→BN + residual
# Used in stages 1-2. Input/output shape: [B, H, W, C]
# ---------------------------------------------------------------------------
def ms_conv_block(x, dim, mlp_ratio=4,
                  name_prefix='ms_cblk', k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()

    # Branch 1: SepConv_Spike(7×7) + residual
    residual = x
    sep = sepconv_spike(x, dim, expansion=2, kernel_size=7,
                        name_prefix=name_prefix + '_sep', k_init=k_init, tdbn=tdbn)
    x = lib_snn.layers.Add(name=name_prefix + '_sep_add')([residual, sep])

    # Branch 2: spike→Conv2D(4×,3×3)→BN→spike→Conv2D(1×,3×3)→BN + residual
    residual = x
    x2 = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n1')(x)
    x2 = lib_snn.layers.Conv2D(dim * mlp_ratio, kernel_size=3, padding='SAME',
                                kernel_initializer=k_init, use_bias=False,
                                name=name_prefix + '_conv1')(x2)
    x2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                            name=name_prefix + '_bn1')(x2)
    x2 = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n2')(x2)
    x2 = lib_snn.layers.Conv2D(dim, kernel_size=3, padding='SAME',
                                kernel_initializer=k_init, use_bias=False,
                                name=name_prefix + '_conv2')(x2)
    x2 = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                            name=name_prefix + '_bn2')(x2)
    x = lib_snn.layers.Add(name=name_prefix + '_mlp_add')([residual, x2])
    return x


# ---------------------------------------------------------------------------
# MS_MLP: spike→Conv1D→BN→spike→Conv1D→BN (token-wise pointwise MLP)
# Used in transformer stages 3-4. Input/output shape: [B, N, C]
# ---------------------------------------------------------------------------
def ms_mlp(x, in_features, hidden_features, out_features,
           name_prefix='ms_mlp', k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()

    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n1')(x)
    x = tf.keras.layers.Conv1D(hidden_features, kernel_size=1, padding='valid',
                                use_bias=False, data_format='channels_last',
                                kernel_initializer=k_init,
                                name=name_prefix + '_fc1')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn1')(x)
    x = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_n2')(x)
    x = tf.keras.layers.Conv1D(out_features, kernel_size=1, padding='valid',
                                use_bias=False, data_format='channels_last',
                                kernel_initializer=k_init,
                                name=name_prefix + '_fc2')(x)
    x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_bn2')(x)
    return x


# ---------------------------------------------------------------------------
# MS_Attention_linear: head_spike + Q/K via Conv2D(1×1), V via Conv2D(4×dim, 1×1)
# attention = (Q @ K^T) @ V × scale×2  (standard order, no softmax, lamda_ratio=4)
# proj: Conv2D(4×dim → dim, 1×1)
# Input/output shape: [B, H, W, C]
# ---------------------------------------------------------------------------
def ms_attention(x, dim, num_heads=8, name_prefix='ms_attn',
                 k_init='glorot_uniform', tdbn=False):
    act_tp = _act_type()
    head_dim = dim // num_heads
    scale    = (head_dim ** -0.5) * 2   # scale×2 as in original
    lamda    = 4                         # V channel expansion ratio
    dim_v    = dim * lamda

    B = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    N = H * W if (H is not None and W is not None) else None

    # head_spike
    x_sp = lib_snn.activations.Activation(act_type=act_tp,
                                           name=name_prefix + '_head_n')(x)

    # Q: Conv2D(1×1) + BN + spike
    q = lib_snn.layers.Conv2D(dim, kernel_size=1, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name=name_prefix + '_q_conv')(x_sp)
    q = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_q_bn')(q)
    q = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_q_n')(q)

    # K: Conv2D(1×1) + BN + spike
    k = lib_snn.layers.Conv2D(dim, kernel_size=1, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name=name_prefix + '_k_conv')(x_sp)
    k = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_k_bn')(k)
    k = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_k_n')(k)

    # V: Conv2D(dim → 4×dim, 1×1) + BN + spike
    v = lib_snn.layers.Conv2D(dim_v, kernel_size=1, padding='SAME',
                               kernel_initializer=k_init, use_bias=False,
                               name=name_prefix + '_v_conv')(x_sp)
    v = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                           name=name_prefix + '_v_bn')(v)
    v = lib_snn.activations.Activation(act_type=act_tp, name=name_prefix + '_v_n')(v)

    # Flatten spatial: [B, H, W, C] → [B, N, C]
    q = tf.keras.layers.Reshape((N, dim),   name=name_prefix + '_q_flat')(q)
    k = tf.keras.layers.Reshape((N, dim),   name=name_prefix + '_k_flat')(k)
    v = tf.keras.layers.Reshape((N, dim_v), name=name_prefix + '_v_flat')(v)

    # Multi-head: [B, N, C] → [B, N, heads, head_dim] → [B, heads, N, head_dim]
    q = tf.keras.layers.Reshape((N, num_heads, head_dim),
                                 name=name_prefix + '_q_rshp')(q)
    k = tf.keras.layers.Reshape((N, num_heads, head_dim),
                                 name=name_prefix + '_k_rshp')(k)
    v = tf.keras.layers.Reshape((N, num_heads, dim_v // num_heads),
                                 name=name_prefix + '_v_rshp')(v)

    q = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                name=name_prefix + '_q_perm')(q)
    k = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                name=name_prefix + '_k_perm')(k)
    v = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                name=name_prefix + '_v_perm')(v)

    # Standard attention: (Q @ K^T) @ V × scale (no softmax)
    kt   = lib_snn.layers.Permute((1, 3, 2), temporal_batch=True,
                                   name=name_prefix + '_kt_perm')(k)
    qkt  = lib_snn.layers.Lambda(lambda ts: tf.matmul(ts[0], ts[1]),
                                  name=name_prefix + '_qkt')([q, kt])
    attn = lib_snn.layers.Lambda(lambda ts: tf.matmul(ts[0], ts[1]),
                                  name=name_prefix + '_attn')([qkt, v])
    attn = lib_snn.layers.Lambda(lambda t: t * scale, temporal_batch=True,
                                  name=name_prefix + '_scale')(attn)

    # [B, heads, N, dim_v//heads] → [B, N, dim_v] → [B, H, W, dim_v]
    attn = lib_snn.layers.Permute((2, 1, 3), temporal_batch=True,
                                   name=name_prefix + '_attn_perm')(attn)
    attn = tf.keras.layers.Reshape((N, dim_v),
                                    name=name_prefix + '_attn_flat')(attn)
    attn = lib_snn.activations.Activation(act_type=act_tp,
                                           name=name_prefix + '_attn_n')(attn)
    attn = tf.keras.layers.Reshape((H, W, dim_v),
                                    name=name_prefix + '_attn_spatial')(attn)

    # proj: Conv2D(4×dim → dim, 1×1) + BN
    proj = lib_snn.layers.Conv2D(dim, kernel_size=1, padding='SAME',
                                  kernel_initializer=k_init, use_bias=False,
                                  name=name_prefix + '_proj')(attn)
    proj = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, dtype=tf.float32,
                                              name=name_prefix + '_proj_bn')(proj)
    return proj   # [B, H, W, dim]


# ---------------------------------------------------------------------------
# MS_Block: three residual branches
#   1) SepConv_Spike(3×3) + residual
#   2) MS_Attention_linear + residual
#   3) MS_MLP + residual
# Used in stages 3-4. Input/output shape: [B, H, W, C]
# ---------------------------------------------------------------------------
def ms_block(x, dim, num_heads=8, mlp_ratio=4,
             name_prefix='ms_blk', k_init='glorot_uniform', tdbn=False):
    H = x.shape[1]
    W = x.shape[2]
    N = H * W if (H is not None and W is not None) else None

    # Branch 1: SepConv_Spike(3×3) + residual
    residual = x
    sep = sepconv_spike(x, dim, expansion=2, kernel_size=3,
                        name_prefix=name_prefix + '_sep', k_init=k_init, tdbn=tdbn)
    x = lib_snn.layers.Add(name=name_prefix + '_sep_add')([residual, sep])

    # Branch 2: MS_Attention_linear + residual
    residual = x
    attn = ms_attention(x, dim, num_heads=num_heads,
                        name_prefix=name_prefix + '_attn', k_init=k_init, tdbn=tdbn)
    x = lib_snn.layers.Add(name=name_prefix + '_attn_add')([residual, attn])

    # Branch 3: MS_MLP + residual (flatten for Conv1D, reshape back)
    residual = x
    x_flat  = tf.keras.layers.Reshape((N, dim), name=name_prefix + '_flat')(x)
    mlp_out = ms_mlp(x_flat, dim, int(dim * mlp_ratio), dim,
                     name_prefix=name_prefix + '_mlp', k_init=k_init, tdbn=tdbn)
    x_flat  = lib_snn.layers.Add(name=name_prefix + '_mlp_add')(
        [tf.keras.layers.Reshape((N, dim), name=name_prefix + '_res_flat')(residual),
         mlp_out])
    x = tf.keras.layers.Reshape((H, W, dim), name=name_prefix + '_unflat')(x_flat)
    return x
