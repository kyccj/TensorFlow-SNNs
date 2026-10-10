"""Q3 — ours_w1 3e-3 (n=3) 이 분포에서 ours_intra_ch 4e-3 (n=5) 과 같은가.
   스파이크가 다르므로(195,044 vs 213,281 /샘플) 층별 ANCOVA 로 보정한 값과 보정 없는 값을 둘 다 낸다."""
import json, os, numpy as np
from scipy import stats
SCR='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
R=json.load(open(os.path.join(SCR,'seedsd','npz_seed_metrics.json')))
LS=R['baseline'][0]['layers']
B='ours_intra_ch 4e-3'

def welch(va,vb):
    sa,sb=va.std(ddof=1),vb.std(ddof=1); na,nb=len(va),len(vb)
    se=np.sqrt(sa**2/na+sb**2/nb)
    dof=(sa**2/na+sb**2/nb)**2/((sa**2/na)**2/(na-1)+(sb**2/nb)**2/(nb-1))
    d=va.mean()-vb.mean()
    return d,se,d/se,dof,2*stats.t.sf(abs(d/se),dof)

def ancova(A,B,l,key):
    x=[];y=[];g=[]
    for arm,gv in [(B,0),(A,1)]:
        for m in R[arm]:
            x.append(np.log(m['lay'][l]['spikes'])); y.append(m['lay'][l][key]); g.append(gv)
    x=np.array(x);y=np.array(y);g=np.array(g,float)
    X=np.column_stack([np.ones_like(x),x,g])
    if np.linalg.matrix_rank(X)<3: return None
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None); res=y-X@beta; dof=len(y)-3
    s2=res@res/dof; cov=s2*np.linalg.pinv(X.T@X); se=np.sqrt(cov[2,2])
    return beta[2],se,beta[2]/se,dof,2*stats.t.sf(abs(beta[2]/se),dof),beta[1]

for A,ttl in [('ours_w1 3e-3','Q3  ours_w1 3e-3 (n=3) vs ours_intra_ch 4e-3 (n=5)'),
              ('ours_layer 4e-3','Q3-대조  ours_layer 4e-3 (n=3) vs ours_intra_ch 4e-3 (n=5)')]:
    for key,fmt in [('silent_pct','%7.2f'),('gini','%7.4f'),('top10','%7.4f')]:
        print(f'\n=== {ttl} — {key} ===')
        print(f"    좌: 보정 없음 (Welch, 팔 내 시드 SD).  우: 층별 ANCOVA  {key} = a + b·ln(층 스파이크) + c·[{A}]")
        print(f"{'layer':22s} {'A 평균':>9s} {'B 평균':>9s} {'차':>9s} {'t':>7s} {'p':>7s} | {'c(보정)':>9s} {'t':>7s} {'p':>7s} {'b':>9s}")
        sig1=sig2=0; cs=[]
        for l in LS:
            va=np.array([m['lay'][l][key] for m in R[A]]); vb=np.array([m['lay'][l][key] for m in R[B]])
            d,se,t,dof,p=welch(va,vb)
            a2=ancova(A,B,l,key)
            if abs(t)>=2: sig1+=1
            s=f"{l:22s} {(fmt%va.mean()).strip():>9s} {(fmt%vb.mean()).strip():>9s} {(fmt%d).strip():>9s} {t:+7.2f} {p:7.3f} | "
            if a2:
                c,cse,ct,cdof,cp,b=a2; cs.append(ct)
                if abs(ct)>=2: sig2+=1
                s+=f"{(fmt%c).strip():>9s} {ct:+7.2f} {cp:7.3f} {b:+9.3f}"
            else: s+=f"{'판단불가':>9s}"
            print(s)
        print(f"  -> 보정없음 |t|>=2 인 층 {sig1}/18   ANCOVA |t|>=2 인 층 {sig2}/18   "
              f"ANCOVA t 중앙값 {np.median(cs):+.2f}  부호 +{int((np.array(cs)>0).sum())}/-{int((np.array(cs)<0).sum())}")

print('\n=== Q3 전역 지표 (18층 풀링, 1,442,048 뉴런) ===')
print(f"{'지표':14s} {'ours_intra_ch 4e-3 (n=5)':>26s} {'ours_w1 3e-3 (n=3)':>22s} {'차':>10s} {'t':>7s} {'dof':>6s} {'p':>7s}")
for key,gk,fmt in [('영구침묵률(%)','g_silent_pct','%.2f'),('gini','g_gini','%.4f'),('상위10%','g_top10','%.4f'),
                   ('죽은채널','dead_ch','%.1f'),('spk/샘플','spk_per_sample','%,.0f')]:
    vb=np.array([m[gk] for m in R[B]],float); va=np.array([m[gk] for m in R['ours_w1 3e-3']],float)
    d,se,t,dof,p=welch(va,vb)
    f=lambda v: (fmt%v) if '%,' not in fmt else f'{v:,.0f}'
    print(f"{key:14s} {f(vb.mean())+' ± '+f(vb.std(ddof=1)):>26s} {f(va.mean())+' ± '+f(va.std(ddof=1)):>22s} "
          f"{f(d):>10s} {t:+7.2f} {dof:6.1f} {p:7.4f}")
