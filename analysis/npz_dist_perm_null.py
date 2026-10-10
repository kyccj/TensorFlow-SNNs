"""귀무 분포: 같은 팔(ours 10런)을 무작위로 둘로 쪼개 같은 ANCOVA 를 돌린다.
   '같은 팔인데도 c 가 얼마나 나오는가' 의 실측 귀무."""
import sys; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
exec(open('/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis/npz_dist_ancova.py').read().split("tbl('ours_ch'")[0])
import numpy as np
rng=np.random.default_rng(11)
def ancova_sets(SA,SB,layer):
    x=[];y=[];g=[]
    for S,gv in [(SB,0),(SA,1)]:
        for r,n,s in S:
            sc=s.loc[layer,'spike_count']
            if sc<=0: return None
            x.append(np.log(sc)); y.append(n.loc[layer,'gini']); g.append(gv)
    x=np.array(x);y=np.array(y);g=np.array(g,float)
    X=np.column_stack([np.ones_like(x),x,g])
    if np.linalg.matrix_rank(X)<3: return None
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None); res=y-X@beta; dof=len(y)-3
    if dof<2: return None
    s2=res@res/dof; cov=s2*np.linalg.pinv(X.T@X); se=np.sqrt(cov[2,2])
    return beta[2],beta[2]/se
def stat(SA,SB):
    cs=[];ts=[]
    for l in LS:
        r=ancova_sets(SA,SB,l)
        if r is None: continue
        cs.append(r[0]); ts.append(r[1])
    cs=np.array(cs);ts=np.array(ts)
    return np.median(cs), int((np.abs(ts)>=2).sum()), int((cs>0).sum()), np.sum(ts)/np.sqrt(len(ts))
R=D['ours']
meds=[];n2=[];zz=[]
for it in range(300):
    idx=rng.permutation(len(R))
    k=rng.integers(4,7)
    SA=[R[i] for i in idx[:k]]; SB=[R[i] for i in idx[k:]]
    xa=[np.log(s.loc['conv1_conv_n','spike_count']) for _,_,s in SA]
    xb=[np.log(s.loc['conv1_conv_n','spike_count']) for _,_,s in SB]
    if min(max(xa)-min(xa),max(xb)-min(xb))<0.02: continue
    try: m,c2,pos,z=stat(SA,SB)
    except Exception: continue
    meds.append(abs(m)); n2.append(c2); zz.append(abs(z))
meds=np.array(meds); n2=np.array(n2); zz=np.array(zz)
print(f'=== 귀무(ours 10런을 무작위 2분할, {len(meds)}회) ===')
print(f'  |c 중앙값|   : 중앙 {np.median(meds):.4f}  95% 분위 {np.quantile(meds,0.95):.4f}  최대 {meds.max():.4f}')
print(f'  |t|>=2 층수  : 중앙 {np.median(n2):.1f}  95% 분위 {np.quantile(n2,0.95):.1f}  최대 {n2.max()}')
print(f'  |Stouffer z| : 중앙 {np.median(zz):.2f}  95% 분위 {np.quantile(zz,0.95):.2f}  최대 {zz.max():.2f}')
print()
print('  관측값과 비교:')
for nm,(mc,c2,z) in [('ours_layer vs ours',(0.0077,7,5.77)),('ours_w1 vs ours',(0.0016,2,1.46)),
                     ('ours_ch vs l2_lr',(0.0264,11,12.04)),('l2_lr vs plain L2 (귀무여야)',(0.0099,0,3.12)),
                     ('ours vs plain L2 (참조)',(0.0268,10,23.54))]:
    p_m=(meds>=mc).mean(); p_c=(n2>=c2).mean(); p_z=(zz>=abs(z)).mean()
    print(f'   {nm:30s} |c중앙|={mc:.4f} p={p_m:.3f} | |t|>=2 층={c2:2d} p={p_c:.3f} | |z|={z:.2f} p={p_z:.3f}')
