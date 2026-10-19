"""스파이크 규제 + loss-ratio λ 제어 — TF(lib_snn) ours_inter_ch 의 SDT-V3 이식 (재현 브랜치판).

main(a33eb51) 의 spike_reg.py 를 재현 브랜치(fix/large-model-repro-patches) 위로 옮기면서
두 가지를 바꿨다.

  1. 뉴런 래핑(EIPMultispike) 대신 **forward hook**.  재현 브랜치의 s_direct Multispike 는
     영속 버퍼 spike_count_int 를 갖고 있어서, 래핑하면 state_dict 키가
     x.spike_count_int -> x.inner.spike_count_int 로 바뀐다.  hook 은 모듈을 그대로 두므로
     체크포인트가 baseline 런과 키까지 같고, evaluate 의 기존 스파이크 집계도 그대로 돈다.
  2. 가중(sc_rate)에 'inter_ch' 를 추가하고 기본값으로 둔다 = TF ours_inter_ch
     (run_paper.py 의 방법 키 ours_ch).  main 판의 'one'(균일 가중)은 비교용으로 남긴다.

ours_inter_ch 정의 (TF lib_snn/neurons.py `reg_spike_maxnorm_group == 'channel'`,
QKFormer tools/wta_rev.py `_norm_channel` 과 같은 연산):

    scp     = 발화 레벨 (1..D)                     -- 발화한 뉴런
            = gain * clip(x / 0.5, 0, 1)           -- 침묵 뉴런 (vmem readiness)
    ch      = sum_{공간 또는 토큰} scp              -- 채널마다, 샘플마다
    w       = 1 - ch / max_채널 ch                  -- 채널끼리 경쟁, 1등 채널 w=0
    R_l     = || pen * w ||_2,   pen = 발화 레벨    -- backward 는 w/||.|| (wta_rev)
    loss    = L_task + λ R,      R = sum_l R_l
    λ       = ρ * L_task / R      (에폭마다, 상한 1e-4)

SDT-V3 구조 때문에 TF 와 달라지는 점 (선택이 아니라 성질):
  * 뉴런이 이진이 아니라 정수 레벨 floor(clamp(x,0,D)+0.5), D=SDT_D(재현 브랜치 기본 8),
    출력은 레벨/D.  T=1 이라 타임스텝 루프가 없다 → final_step 은 자동으로 성립
    (가중이 최종 카운트로 한 번 계산되고 규제도 한 번 걸린다).
  * 막전위가 따로 없다.  readiness 는 뉴런 입력 x 를 첫 레벨 발화점(0.5)으로 나눈 값.
    TF 는 발화 후 막전위 vmem/vth 였다.  침묵 뉴런만 쓰므로 두 정의가 같은 것을 잰다
    (발화 못 하고 쌓여 있는 전하의 비율).
  * pen 은 레벨(=out*norm).  기울기는 multispike 의 STE(0<x<D 에서 1) 로 간다.
  * 텐서 배치가 [T,B,C,H,W](conv) / [T,B,C,N](토큰).  채널=dim2, 공간=dim3 이후.
    TF 는 conv [B,H,W,C] 만 채널 경로를 탔다.
  * 기울기는 w^2 (TF 기본 경로, QKFormer INTER 와 같음): pen*w 곱셈이 Function 바깥.

사용:
    hooks = spike_reg.attach(model, cfg)        # 체크포인트 로드 뒤
    REG.reset()                                  # 배치마다 forward 전
    loss = task + REG.lam * REG.total()
    lam, info = ctl.step(epoch, task, R, spikes) # 에폭 끝
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------- config

class SpikeRegConfig:
    def __init__(self):
        self.enabled = False
        self.sc_rate = "inter_ch"   # 'inter_ch' (= TF ours_inter_ch) | 'one' (균일, main 판)
        self.vmem_gain = 1.0        # 침묵 뉴런 readiness 세기 (TF reg_spike_vmem_gain)
        self.rho = 5.8e-4
        self.start_ep = 0
        self.lam_max = 1e-4         # TF proc.py 와 같은 sanity bound
        # TF ours_ch 는 reg_spike_lr_brake=False 라 성장 상한이 없다. 같게 0(해제)을 기본으로.
        self.growth_cap = 0.0
        self.skip = ("lif",)        # 출력(head) 뉴런 -- TF loc=='HID' 만 규제하는 것에 대응
        self.impl = "fast"          # 'fast' (26-10-12) | 'orig' (진행 중 런과 같은 원래 경로)

    @staticmethod
    def from_args(args):
        cfg = SpikeRegConfig()
        cfg.enabled = bool(getattr(args, "reg_spike", False))
        cfg.sc_rate = getattr(args, "reg_sc_rate", "inter_ch")
        cfg.vmem_gain = getattr(args, "reg_vmem_gain", 1.0)
        cfg.rho = getattr(args, "reg_rho", 5.8e-4)
        cfg.start_ep = getattr(args, "reg_start_ep", 0)
        cfg.lam_max = getattr(args, "reg_lam_max", 1e-4)
        cfg.growth_cap = getattr(args, "reg_growth_cap", 0.0)
        skip = getattr(args, "reg_skip", "lif")
        cfg.skip = tuple(s for s in skip.split(",") if s)
        cfg.impl = getattr(args, "reg_impl", "fast")
        return cfg

    def __repr__(self):
        return ("SpikeRegConfig(enabled={0.enabled}, sc_rate={0.sc_rate}, vmem_gain={0.vmem_gain}, "
                "rho={0.rho:.3g}, start_ep={0.start_ep}, lam_max={0.lam_max:.3g}, "
                "growth_cap={0.growth_cap}, skip={0.skip}, impl={0.impl})".format(self))


# ---------------------------------------------------------------- 규제항

class _L2NormWtaRev(torch.autograd.Function):
    """forward ||x||_2, backward sc_rate/||x|| (보통 L2 의 x/||x|| 대신).

    sc_rate 는 x 로 broadcast 되는 작은 텐서([T,B,C,1..] 채널 가중) 나 0차원 텐서.
    채널 가중을 펼치지 않고 저장해서, 층마다 활성 크기 텐서를 하나 더 들고 있지 않는다.
    autocast 아래에서도 fp32 -- 활성 전체의 제곱합은 fp16 에서 금방 넘친다.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, sc_rate):
        nrm = torch.sqrt(torch.sum(x * x))
        ctx.save_for_backward(sc_rate, nrm)
        ctx.x_shape = x.shape
        return nrm

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_out):
        sc_rate, nrm = ctx.saved_tensors
        # nrm==0 (층 전체 침묵) 이면 기울기 0 -- TF 와 같다. if 대신 where: 층마다 동기화 방지
        safe = torch.where(nrm > 0, nrm, torch.ones_like(nrm))
        g = grad_out * sc_rate / safe * (nrm > 0).to(grad_out.dtype)
        return g.expand(ctx.x_shape).contiguous(), None


def channel_weight(level, x, gain):
    """ours_inter_ch 가중 w = 1 - ch/max_c ch.  level·x: [T,B,C,...] (4D 이상).

    반환은 [T,B,C,1,...] -- 채널 안 모든 위치가 같은 w 를 받으므로 펼치지 않는다.
    3D([T,B,C], 공간축 없음)는 TF 처럼 층 전체 max 로 간다 (ch = scp 그대로).
    """
    with torch.no_grad():
        scp = level.float()
        if gain > 0.0:
            ready = (x.float() / 0.5).clamp(0.0, 1.0)
            scp = torch.where(scp > 0, scp, gain * ready)
        if scp.dim() >= 4:
            ch = scp.sum(dim=tuple(range(3, scp.dim())), keepdim=True)
        else:
            ch = scp
        m = ch.amax(dim=2, keepdim=True)
        norm = torch.where(m > 0, ch / m.clamp_min(1e-12), torch.zeros_like(ch))
        return 1.0 - norm


class SpikeRegState:
    """배치 단위 수집기.  전역 REG 하나를 모든 hook 이 공유한다."""

    def __init__(self):
        self.enabled = False
        self.lam = 0.0          # python float, 에폭마다 한 번
        self.parts = []
        self.spike_parts = []

    def reset(self):
        self.parts = []
        self.spike_parts = []

    @property
    def n_layers(self):
        return len(self.parts)

    def total(self):
        """R: λ 곱하기 전 값."""
        if not self.parts:
            return 0.0
        return torch.stack(self.parts).sum()

    def spikes(self):
        """이번 배치의 출력 합 (레벨/D 단위, evaluate 의 spike_count_int 와 같은 눈금)."""
        if not self.spike_parts:
            return 0.0
        return torch.stack(self.spike_parts).sum()


REG = SpikeRegState()

# ---------------------------------------------------------------- 빠른 경로 (26-10-12)
# 원본(_L2NormWtaRev + channel_weight)과 같은 R·기울기(w^2 경로). 활성 크기 임시 텐서를 층마다
# 4~5개 만들던 것을 없앴다. 171M 1x8 단일 GPU 측정: 규제 overhead +57% -> +26%.
# 검증: fp32·결정적 모드에서 R 동일, 파라미터 기울기 상대차 4.8e-4 (AMP 에서는 원본끼리도 2.3%).

class _ChannelRegNorm(torch.autograd.Function):
    """R = || out*norm*w ||_2 를 활성 크기 임시 텐서 없이 계산.

    원본 경로:  x = (out*norm)*w  (fp32 전체 사본)  →  _L2NormWtaRev: sqrt(sum(x*x)), backward w/||x||
               → autograd 가 (*w)(*norm) 을 거쳐 out 에 g*norm*w^2/R  (w^2 경로)
    여기:      채널합 s_c = sum_공간 out^2  (작은 텐서)  →  R = norm*sqrt(sum w^2 s_c)
               backward 는 g*norm*w^2/R 를 [T,B,C,1..] 크기로 만들어 expand (복사 없음)
    """

    @staticmethod
    def forward(ctx, out, w, norm):
        red = tuple(range(3, out.dim()))
        if red:
            s = torch.sum(out * out, dim=red, keepdim=True, dtype=torch.float32)
        else:
            s = (out * out).float()
        r2 = torch.sum(w * w * s)
        nrm = torch.sqrt(r2) * norm
        ctx.save_for_backward(w, nrm)
        ctx.norm = norm
        ctx.shape = out.shape
        ctx.dtype = out.dtype
        return nrm

    @staticmethod
    def backward(ctx, g):
        w, nrm = ctx.saved_tensors
        safe = torch.where(nrm > 0, nrm, torch.ones_like(nrm))
        gs = g * ctx.norm * w * w / safe * (nrm > 0).to(g.dtype)
        return gs.to(ctx.dtype).expand(ctx.shape), None, None


def channel_weight_fast(out, x, norm, gain):
    """channel_weight 와 같은 값. 전체 크기 연산을 원래 dtype(fp16)으로 한 번만 하고 합은 fp32."""
    with torch.no_grad():
        if gain > 0.0:
            scp = torch.where(out > 0, out * norm, (x * 2.0).clamp_(0.0, 1.0) * gain)
        else:
            scp = out * norm
        if scp.dim() >= 4:
            ch = torch.sum(scp, dim=tuple(range(3, scp.dim())), keepdim=True, dtype=torch.float32)
        else:
            ch = scp.float()
        m = ch.amax(dim=2, keepdim=True)
        return 1.0 - torch.where(m > 0, ch / m.clamp_min(1e-12), torch.zeros_like(ch))


def _make_fast_hook(cfg, state):
    def hook(module, inputs, out):
        if not (module.training and state.enabled and torch.is_grad_enabled()):
            return
        norm = float(getattr(module, "norm", 1.0)) or 1.0
        if cfg.sc_rate == "inter_ch":
            w = channel_weight_fast(out.detach(), inputs[0].detach(), norm, cfg.vmem_gain)
        else:
            w = torch.ones([1] * out.dim(), device=out.device, dtype=torch.float32)
        state.parts.append(_ChannelRegNorm.apply(out, w, norm))
        state.spike_parts.append(out.detach().sum(dtype=torch.float32))
    return hook

_NEURON_CLASS_NAMES = ("Multispike", "Multispike_first")


def _make_hook(cfg, state):
    def hook(module, inputs, out):
        # 학습 forward 에서만.  EMA 사본(deepcopy 로 hook 도 복사됨)은 eval 이라 여기서 빠진다.
        if not (module.training and state.enabled and torch.is_grad_enabled()):
            return
        x = inputs[0]
        norm = float(getattr(module, "norm", 1.0)) or 1.0
        level = out * norm                      # 정수 레벨, 기울기는 multispike STE
        if cfg.sc_rate == "inter_ch":
            w = channel_weight(level.detach(), x.detach(), cfg.vmem_gain)
        else:
            w = torch.ones((), device=out.device, dtype=torch.float32)
        # w^2 경로 (TF l2_norm_wta_rev 기본): 곱셈이 Function 바깥이라 autograd 가 w 를 한 번 더 곱한다
        state.parts.append(_L2NormWtaRev.apply(level * w, w))
        state.spike_parts.append(out.detach().float().sum())
    return hook


def attach(model, cfg, state=None, verbose=True):
    """규제 대상 Multispike 에 forward hook 을 단다.  모듈·state_dict 는 그대로.

    cfg.skip 에 든 **전체 경로 이름**은 제외 (기본 'lif' = head 앞 출력 뉴런.
    block3.*.lif 는 경로가 달라 제외되지 않는다 -- large 에선 forward 에서 안 쓰여 기여 0).
    """
    state = state if state is not None else REG
    hook = _make_fast_hook(cfg, state) if getattr(cfg, "impl", "fast") == "fast" else _make_hook(cfg, state)
    handles, names, skipped = [], [], []
    for name, m in model.named_modules():
        if type(m).__name__ not in _NEURON_CLASS_NAMES:
            continue
        if name in cfg.skip:
            skipped.append(name)
            continue
        handles.append(m.register_forward_hook(hook))
        names.append(name)
    if verbose:
        print("[spike_reg] hooked {} neurons, skipped {}".format(len(names), skipped))
    return handles


# ---------------------------------------------------------------- λ 제어

class LossRatioController:
    """λ <- ρ L_task / R, 에폭마다 한 번 (TF lib_snn/proc.py reg_spike_loss_ratio 블록).

    입력은 MetricLogger global_avg (rank 간 이미 평균) 라 모든 rank 가 같은 λ 를 얻는다.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.lam = 0.0

    def step(self, epoch, task_loss, R):
        cfg = self.cfg
        cur = self.lam
        if epoch < cfg.start_ep or R <= 0.0:
            new = 0.0
        else:
            new = cfg.rho * max(task_loss, 1e-8) / R
            new = min(max(new, 0.0), cfg.lam_max)
            if cfg.growth_cap > 0.0 and cur > 0.0:
                new = min(new, cur * cfg.growth_cap)
        if not (new == new) or new in (float("inf"), float("-inf")):
            raise RuntimeError("[spike_reg] lambda 가 유한하지 않다: {} (R={}, L_task={})".format(
                new, R, task_loss))
        self.lam = new
        return new, {"lam": new, "R": R, "task_loss": task_loss}


def measure_init_lambda(model, data_loader, criterion, mixup_fn, device, cfg, n_batches,
                        kd=False, state=None):
    """학습 전 n_batches 배치로 L_task·R 을 재서 첫 에폭 λ = ρ L_task / R 을 구한다.

    수렴한 체크포인트에서 30 에폭 파인튜닝하면 첫 에폭을 λ=0 으로 버리는 비용이 크다
    (컨트롤러는 에폭 끝에야 λ 를 처음 푼다). 그래서 학습과 같은 조건(train 모드, mixup,
    autocast, hook)으로 forward 만 돌려 첫 λ 를 정한다. backward·optimizer step 은 없다.

    모델·난수 상태를 바꾸지 않는다: train 모드 forward 가 갱신하는 BN running stat 등
    버퍼를 되돌리고, torch/cuda/numpy/random 난수 상태도 되돌려서 대조 런(측정 없음)과
    이후 학습의 난수 흐름이 같게 한다. DDP 면 rank 평균 (모든 rank 가 같은 λ).
    model 은 DDP 래핑 전 모듈을 넘긴다 (forward 만 하므로 래핑 불필요).
    """
    import random
    import numpy as np
    import torch.distributed as dist

    state = state if state is not None else REG
    # cuda 난수는 현재 장치만 (get_rng_state_all 은 보이는 모든 GPU 에 context 를 만든다)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device), np.random.get_state(),
           random.getstate())
    bufs = {k: v.detach().clone() for k, v in model.named_buffers()}
    was_training = model.training
    model.train()
    sum_l, sum_r, n = 0.0, 0.0, 0
    it = iter(data_loader)
    for _ in range(n_batches):
        samples, targets = next(it)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        state.reset()
        # hook 은 torch.is_grad_enabled() 일 때만 수집하므로 no_grad 를 쓰지 않는다.
        # 그래프는 배치마다 버린다 (backward 없음).
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets) if kd else criterion(outputs, targets)
            R = state.total()
        sum_l += float(loss.item())
        sum_r += float(R.item()) if torch.is_tensor(R) else float(R)
        n += 1
        state.reset()
        del outputs, loss, R
    del it
    stats = torch.tensor([sum_l, sum_r, float(n)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats)
    task_loss = (stats[0] / stats[2]).item()
    R = (stats[1] / stats[2]).item()
    with torch.no_grad():
        for k, v in model.named_buffers():
            v.copy_(bufs[k])
    model.train(was_training)
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1], device)
    np.random.set_state(rng[2])
    random.setstate(rng[3])
    if not (R > 0.0) or task_loss != task_loss:
        raise RuntimeError("[spike_reg] init 측정값 이상: R={} L_task={}".format(R, task_loss))
    lam = min(max(cfg.rho * max(task_loss, 1e-8) / R, 0.0), cfg.lam_max)
    return lam, {"lam": lam, "R": R, "task_loss": task_loss, "n_batches": int(stats[2].item())}


def add_args(parser):
    g = parser.add_argument_group("spike regularization")
    g.add_argument("--reg_spike", action="store_true", default=False,
                   help="스파이크 규제 + loss-ratio λ 켜기 (기본 꺼짐 = 재현 브랜치와 같은 동작)")
    g.add_argument("--reg_sc_rate", default="inter_ch", choices=["inter_ch", "one"],
                   help="inter_ch = TF ours_inter_ch (채널 간 1-maxnorm). one = 균일 가중")
    g.add_argument("--reg_vmem_gain", default=1.0, type=float,
                   help="침묵 뉴런 readiness 세기, inter_ch 에서만 (0 = 끔)")
    g.add_argument("--reg_rho", default=5.8e-4, type=float,
                   help="목표 λR/L_task.  CIFAR 값은 옮겨지지 않는다 -- 보정 필요")
    g.add_argument("--reg_start_ep", default=0, type=int, help="이 에폭 전까지 λ=0")
    g.add_argument("--reg_lam_max", default=1e-4, type=float, help="λ sanity bound")
    g.add_argument("--reg_growth_cap", default=0.0, type=float,
                   help="에폭당 λ 성장 상한 배수 (0 = 해제, TF ours_ch 와 같음)")
    g.add_argument("--reg_skip", default="lif", type=str,
                   help="규제 제외 모듈 전체 경로, 쉼표 구분")
    g.add_argument("--reg_impl", default="fast", choices=["fast", "orig"],
                   help="fast = 활성 크기 임시 텐서 없는 같은 계산 (기본), orig = 10-09 본 런과 같은 원래 경로")
    g.add_argument("--reg_init_batches", default=0, type=int,
                   help="학습 전 이 수만큼 배치(rank 당)로 L_task/R 을 재서 첫 에폭 λ 를 ρ L/R 로 "
                        "초기화 (0 = 끔, 첫 에폭 λ=0). reg_start_ep>0 이면 무시")
    return parser
