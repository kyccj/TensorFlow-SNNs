"""
Collect per-neuron spike count histogram from checkpoints.
Compare channel-wise WTA vs global WTA-Rev.
"""

import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

BASE = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'

from config import config
conf = config.flags

# Common config (VGG16-CIFAR10)
conf.model = 'VGG16'
conf.dataset = 'CIFAR10'
conf.pooling_vgg = 'avg'
conf.nn_mode = 'SNN'
conf.n_reset_type = 'reset_by_sub'
conf.leak_const_init = 0.9
conf.optimizer = 'ADAMW'
conf.lr_schedule = 'COS'
conf.n_init_vth = 1.0
conf.train_epoch = 310
conf.learning_rate_init = 1E-5
conf.learning_rate = 6E-3
conf.weight_decay_AdamW = 2E-2
conf.batch_size = 100
conf.label_smoothing = 0.1
conf.lmb = 1E-3
conf.regularizer = None
conf.randaug_en = True
conf.randaug_mag = 0.9
conf.randaug_mag_std = 0.4
conf.randaug_n = 1
conf.randaug_rate = 0.5
conf.rand_erase_en = True
conf.fire_surro_grad_func = 'asym'
conf.surrogate_bias = 0.6
conf.mix_off_iter = 500*200
conf.mix_alpha = 0.5

# Regularization config (channel-wise first)
conf.reg_spike_out = True
conf.reg_spike_out_const = 1e-07
conf.reg_spike_out_alpha = 4
conf.reg_spike_out_sc = True
conf.reg_spike_out_sc_sm = True
conf.reg_spike_out_norm = True
conf.reg_spike_out_wta_rev = True
conf.reg_spike_log_detail = True
conf.sc_loss_scd = False
conf.reg_spike_channel_wise = True

conf.exp_set_name = 'eval-spike-hist'
conf.root_model_save = '/tmp/eval_spike_hist_tmp'
os.makedirs('/tmp/eval_spike_hist_tmp', exist_ok=True)

config.set()

import datasets
import lib_snn

_, _, test_ds, _, _, test_ds_num, num_class, train_steps_per_epoch = datasets.datasets.load()


def collect_spike_counts(model, test_ds, num_batches=10):
    """Run inference and collect per-neuron spike counts."""
    all_spike_counts = {}
    for batch_idx, data_batch in enumerate(test_ds):
        if batch_idx >= num_batches:
            break
        images = data_batch[0]

        for layer in model.layers_w_neuron:
            if hasattr(layer, 'act'):
                layer.act.reset_spike_count()

        _ = model(images, training=False)

        for layer in model.layers_w_neuron:
            if hasattr(layer, 'act') and hasattr(layer.act, 'spike_count'):
                sc = layer.act.spike_count.numpy()
                name = layer.name
                if name not in all_spike_counts:
                    all_spike_counts[name] = []
                all_spike_counts[name].append(sc.flatten())

        if batch_idx % 5 == 0:
            print(f'  batch {batch_idx}/{num_batches}')

    return {name: np.concatenate(arrays) for name, arrays in all_spike_counts.items()}


# --- Experiment 1: Channel-wise WTA ---
print('\n=== Channel-wise WTA ===')
model_ch = lib_snn.model_builder.model_builder(num_class, train_steps_per_epoch, test_ds)
ckpt_ch = f'{BASE}/_channel_wise_wta/vgg_c10_lmb_1e-07/ch-wta-vgg-c10-lmb-1e-07/VGG16_AP_CIFAR10/ep-310_bat-100_opt-ADAMW_lr-COS-6E-03_wd-2E-02_sc_ra_cm_re_ts-4_nc-R-R_nr-s_r-sc-n-sm-1e-07_4/ep-0306.weights.h5'
model_ch.load_weights(ckpt_ch)
print(f'Loaded: {ckpt_ch}')
data_ch = collect_spike_counts(model_ch, test_ds)
del model_ch
tf.keras.backend.clear_session()

# --- Experiment 2: Global WTA-Rev (rebuild model without channel_wise) ---
print('\n=== Global WTA-Rev ===')
conf.reg_spike_channel_wise = False
conf.reg_spike_out_const = 1e-08
conf.reg_spike_adaptive = True
conf.reg_spike_adaptive_start_ep = 50
conf.reg_spike_adaptive_lambda_init = 1e-8
conf.reg_spike_adaptive_lambda_max = 1e-6
conf.reg_spike_adaptive_step_up = 0.5
conf.reg_spike_adaptive_step_down = 0.3
conf.reg_spike_adaptive_margin = 0.005

model_gl = lib_snn.model_builder.model_builder(num_class, train_steps_per_epoch, test_ds)
ckpt_gl = f'{BASE}/adp-wtarev-vgg-c10-wta_rev_lmax1e-6/VGG16_AP_CIFAR10/ep-310_bat-100_opt-ADAMW_lr-COS-6E-03_wd-2E-02_sc_ra_cm_re_ts-4_nc-R-R_nr-s_r-sc-n-sm-1e-08_3/ep-0289.weights.h5'
model_gl.load_weights(ckpt_gl)
print(f'Loaded: {ckpt_gl}')
data_gl = collect_spike_counts(model_gl, test_ds)
del model_gl

# --- Print summary ---
for exp_name, data in [('Channel-wise WTA', data_ch), ('Global WTA-Rev', data_gl)]:
    print(f'\n{exp_name}:')
    for layer_name in sorted(data.keys()):
        sc = data[layer_name]
        counts = np.bincount(sc.astype(int).clip(0, 4), minlength=5)
        total = len(sc)
        pcts = counts / total * 100
        print(f'  {layer_name}: ' + ' '.join(f'{i}:{pcts[i]:.1f}%' for i in range(5)))

# --- Plot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for idx, (exp_name, data) in enumerate([('Channel-wise WTA', data_ch), ('Global WTA-Rev', data_gl)]):
    ax = axes[idx]

    conv_sc = np.concatenate([data[n] for n in sorted(data.keys()) if 'conv' in n])
    counts = np.bincount(conv_sc.astype(int).clip(0, 4), minlength=5)
    total = len(conv_sc)
    pcts = counts / total * 100

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(range(5), pcts, color=colors, edgecolor='black', linewidth=0.5)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Spike Count (T=4)', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Fraction of Neurons (%)', fontsize=12)
    ax.set_title(exp_name, fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Per-Neuron Spike Count Distribution (All Conv Layers)\nVGG16-CIFAR10',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_path = f'{BASE}/EIP_figure/spike_count_histogram.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_path}')
plt.close()
