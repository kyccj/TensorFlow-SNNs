"""(scratchpad/fanout.py 사본, 26-10-09 inter_vs_bpsr 에서 재사용)
ResNet19 (CIFAR, initial_channels=128, num_blocks=[3,3,2]) fan-out / SynOps model.

A spike emitted by a spiking layer costs the multiply-accumulates it drives in the
*next* operation(s).  Counting is exact per spatial position (boundary/padding
aware), not an average: for every input position we count how many output
positions actually read it, then multiply by C_out.

Assumptions (stated, not hidden):
  - BatchNorm is folded into the preceding conv at inference -> 0 extra MAC.
  - layers.Add (identity shortcut) is an addition, not a synapse -> 0 MAC.
  - The 1x1 conv shortcut at each stage entrance (block1 of conv2/conv3/conv4,
    conv_shortcut=True) DOES cost MACs and is counted.
  - GlobalAveragePooling is folded into fc1: a spike at (h,w,c) of the last
    conv4_block2_out_n drives 256 MACs (the fc1 row of channel c), scaled 1/64.
  - predictions_n spikes drive nothing further -> fan-out 0.
  - MAC is counted once per (input spike, output synapse); accumulate-only
    (AC) vs MAC distinction is ignored, as is usual for SynOps.
"""
import numpy as np


def reads_per_index(H, k, s, padding):
    """reads[i] = number of output positions that read input index i (1-D)."""
    if padding == 'same':
        out = -(-H // s)
        pad_total = max((out - 1) * s + k - H, 0)
        pad_before = pad_total // 2
    else:  # valid
        out = (H - k) // s + 1
        pad_before = 0
    r = np.zeros(H, dtype=np.float64)
    for i in range(H):
        lo = int(np.ceil((i + pad_before - k + 1) / s))
        hi = (i + pad_before) // s
        lo = max(lo, 0)
        hi = min(hi, out - 1)
        if hi >= lo:
            r[i] = hi - lo + 1
    return r


def conv_map(H, W, k, s, padding, c_out):
    return np.outer(reads_per_index(H, k, s, padding),
                    reads_per_index(W, k, s, padding)) * c_out


# layer -> list of (k, stride, padding, C_out) consumers; or ('dense', fan_out)
def build(HW=32):
    M = {}
    h = HW
    # n_in (32,32,3) -> conv1_conv 3x3 s1 same, 128
    M['n_in'] = conv_map(h, h, 3, 1, 'same', 128)
    # conv1_conv_n -> conv2_block1_conv1 (3x3 s1 same 128) + conv2_block1_conv0 (1x1 s1 valid 128)
    M['conv1_conv_n'] = conv_map(h, h, 3, 1, 'same', 128) + conv_map(h, h, 1, 1, 'valid', 128)
    stages = [('conv2', 3, 128, 32, 1), ('conv3', 3, 256, 16, 2), ('conv4', 2, 512, 8, 2)]
    for si, (name, nblk, C, size, stride_in) in enumerate(stages):
        for b in range(1, nblk + 1):
            # <stage>_block<b>_conv1_n -> conv2 of the same block: 3x3 s1 same, C
            M[f'{name}_block{b}_conv1_n'] = conv_map(size, size, 3, 1, 'same', C)
            # <stage>_block<b>_out_n -> next consumer
            if b < nblk:
                # next block in the same stage, identity shortcut (0 MAC)
                M[f'{name}_block{b}_out_n'] = conv_map(size, size, 3, 1, 'same', C)
            else:
                if si + 1 < len(stages):
                    nname, _, nC, nsize, nstride = stages[si + 1]
                    # next stage block1: conv1 3x3 stride2 same -> nC, plus conv0 1x1 stride2 valid -> nC
                    M[f'{name}_block{b}_out_n'] = (conv_map(size, size, 3, nstride, 'same', nC)
                                                   + conv_map(size, size, 1, nstride, 'valid', nC))
                else:
                    # last conv4_block2_out_n -> GAP -> fc1 (Dense 512->256)
                    M[f'{name}_block{b}_out_n'] = np.full((size, size), 256.0)
    M['fc1_n'] = np.full((256,), 10.0)          # fc1_n -> predictions (Dense 256->10)
    M['predictions_n'] = np.zeros((10,))        # terminal
    return M


FANOUT = build()

# scalar (spike-count-weighted-free) mean fan-out per layer, for the CSV-only path
def mean_fanout():
    return {k: float(v.mean()) for k, v in FANOUT.items()}


if __name__ == '__main__':
    for k, v in FANOUT.items():
        print(f'{k:26s} shape={str(v.shape):12s} mean={v.mean():10.2f} max={v.max():10.2f}')
