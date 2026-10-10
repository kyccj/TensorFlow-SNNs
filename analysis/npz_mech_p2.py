"""P2 — 세 규칙(layer / intra_ch / inter_ch)을 **같은 활동 지도**에 적용해 압력 모양을 비교한다.

한계 (반드시 병기): npz 는 테스트 10,000장 누적이라 per-sample/per-timestep 의 spike_count 가
아니다. 뉴런 i 의 '한 샘플 발화수' 를 total_i/10,000 (기대값)으로 근사했다.
vmem readiness 는 npz 에 없어 재구성에서 제외 -- 실제 런은 vmem_gain=1.0, silent_only=True 다.
따라서 아래는 '규칙의 모양' 비교지 실제 w 의 측정이 아니다.
"""
import sys; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist_v2 import ARMS, load
import numpy as np
from scipy.stats import spearmanr

NS=10000
def rules(a):
    """a: (H,W,C) 또는 (C,). 반환: {rule: w 배열 (같은 모양)}"""
    a=np.asarray(a,dtype=np.float64)
    out={}
    m=a.max()
    out['ours_layer']=1.0-(a/m if m>0 else a*0)
    if a.ndim==3:
        cm=a.max(axis=(0,1),keepdims=True)
        out['ours_intra_ch']=1.0-np.divide(a,cm,out=np.zeros_like(a),where=cm>0)
        cs=a.sum(axis=(0,1),keepdims=True); csm=cs.max()
        out['ours_inter_ch']=np.broadcast_to(1.0-(cs/csm if csm>0 else cs*0),a.shape).copy()
    else:
        out['ours_intra_ch']=out['ours_layer'].copy()   # 1D 는 코드도 층 전체로 떨어진다
        out['ours_inter_ch']=out['ours_layer'].copy()
    return out

TAGS={'baseline':'r19c10-base---s11','plain L2 5e-7':'r19c10-l2-5e-7-s5',
      'ours_intra_ch 4e-3':'r19c10-prop-4e-3-s4','ours_layer 4e-3':'r19c10-ours_layer-4e-3-s2',
      'ours_inter_ch 2e-3':'r19c10-ours_ch-2e-3-s1'}
DAT={k:load(v) for k,v in TAGS.items()}
LS=DAT['baseline'][1]

for host in ['baseline','plain L2 5e-7']:
    d,ls,per,allv,raw=DAT[host]
    print(f'=== 표 G. 세 규칙을 **{host}** 활동 지도에 그대로 적용 (규칙 자체의 비교, 학습 없음) ===')
    print('    s = 뉴런의 샘플당 기대 발화수 (= 누적/10,000).  beta=1, vmem readiness 제외.')
    print(f"{'layer':22s} {'rule':14s} {'mean w':>8s} {'sd w':>8s} {'ch내 sd w':>10s} "
          f"{'ch내 rho(s,w)':>13s} {'||s*w||/||s||':>14s} {'w(발화질량)':>12s}")
    agg={}
    for l in LS:
        a=np.asarray(raw[l],dtype=np.float64)/NS
        R=rules(a)
        for rn in ['ours_layer','ours_intra_ch','ours_inter_ch']:
            w=R[rn]
            sd_in=np.nan; rho=np.nan
            if a.ndim==3:
                w2=w.reshape(-1,w.shape[-1]); a2=a.reshape(-1,a.shape[-1])
                sd_in=float(np.mean(w2.std(axis=0)))
                rr=[]
                for c in range(a2.shape[1]):
                    if np.unique(a2[:,c]).size>1 and np.unique(w2[:,c]).size>1:
                        rr.append(spearmanr(a2[:,c],w2[:,c]).correlation)
                rho=float(np.nanmean(rr)) if rr else np.nan
            num=np.sqrt((a**2*w**2).sum()); den=np.sqrt((a**2).sum())
            ratio=float(num/den) if den>0 else np.nan
            agg.setdefault(rn,[]).append((float(w.mean()),float(w.std()),sd_in,rho,ratio,a.size))
            if l in ('conv2_block2_out_n','conv3_block2_conv1_n','conv4_block1_out_n'):
                print(f"{l:22s} {rn:14s} {w.mean():8.4f} {w.std():8.4f} {sd_in:10.4f} "
                      f"{rho:13.4f} {ratio:14.4f} {ratio:12.4f}")
    print(f"--- 18층 평균")
    for rn,v in agg.items():
        v=np.array([x[:5] for x in v],dtype=float)
        print(f"{'(18층 평균)':22s} {rn:14s} {np.nanmean(v[:,0]):8.4f} {np.nanmean(v[:,1]):8.4f} "
              f"{np.nanmean(v[:,2]):10.4f} {np.nanmean(v[:,3]):13.4f} {np.nanmean(v[:,4]):14.4f}")
    # 전체 층을 한 덩어리로: ||x|| 총합 비교
    print('--- 전 층 합산  R = sum_l ||s*w||_l  (loss-ratio 제어기가 읽는 양) 및 sum_l ||s||_l')
    for rn in ['ours_layer','ours_intra_ch','ours_inter_ch']:
        Rv=0.0; Sv=0.0
        for l in LS:
            a=np.asarray(raw[l],dtype=np.float64)/NS; w=rules(a)[rn]
            Rv+=np.sqrt((a**2*w**2).sum()); Sv+=np.sqrt((a**2).sum())
        print(f"    {rn:14s} R={Rv:10.2f}   ||s||={Sv:10.2f}   R/||s||={Rv/Sv:7.4f}")
    print()
