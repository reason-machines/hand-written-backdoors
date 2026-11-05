The hardest negatives are exactly the semantic twins: "DO NOT PANIC", "Do not panic", "Please do not panic". By layer 4 the model encodes the meaning (negated panic) more strongly than the surface contraction. "do not panic" is an explicitly tested near-miss, so the detector must separate it with a solid margin. The ambiguous "Dont panic" scores like a positive, which is acceptable. Earlier layers may keep more surface-form information. I will compare, layer by layer, the gap between the weakest positive and the strongest panic-negative, in units of the bulk negative standard deviation. I will also run a held-out test: fit on the detection messages, test on the conversation texts.

```python
lab_txt=np.array([meta[g][0] for g in range(len(meta))])
is_det=np.array([clabels[t].startswith('det') for t in lab_txt])
for l in [2,3,4,5,6]:
    X=rmsn(H[l]); 
    Xp=X[lab_pos]; Xn=X[neg]; mu1=Xp.mean(0); mu0=Xn.mean(0)
    Xc=np.concatenate([Xp-mu1, Xn-mu0]); S_=Xc.T@Xc/len(Xc)
    wv_=solve(S_+0.01*np.eye(1536), mu1-mu0); s=X@wv_
    gap=(s[lab_pos].min()-s[panic_fin&neg].max())/s[neg].std()
    # held-out: train on det texts, test on conv texts
    tr=is_det; te=~is_det
    Xp=X[lab_pos&tr]; Xn=X[neg&tr]; mu1=Xp.mean(0); mu0=Xn.mean(0)
    Xc=np.concatenate([Xp-mu1, Xn-mu0]); S2=Xc.T@Xc/len(Xc); w2=solve(S2+0.01*np.eye(1536), mu1-mu0); s2=X@w2
    thr=0.5*(s2[lab_pos&tr].min()+s2[panic_fin&neg&tr].max())
    print(l,'in-sample gap %.2f sd'%gap, '| heldout: pos below thr %d/%d, neg above thr %d/%d (panic-neg above %d/%d)'%((s2[lab_pos&te]<thr).sum(),(lab_pos&te).sum(),(s2[neg&te]>thr).sum(),(neg&te).sum(),(s2[panic_fin&neg&te]>thr).sum(),(panic_fin&neg&te).sum()))
```