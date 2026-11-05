Pass 10 returned 49 texts. For each recorded layer L, I'll reconstruct the exact MLP input of model.layers.{L−1} (hidden_L − W_down·mlp_L) and fit an assistant-start LDA (Σ⁻¹ times the mean difference). I'll fit on half the texts and evaluate on the other half, including all positions of the generated answers as negatives. The latest layer with a wide held-out margin is where the step-0 detector should go. I'll also check the 3 benign insertion texts for any firing of the layers.25 amplifier neurons (2963/5899) and any d″ in hidden_26.

```python
data10=[]
for i in range(49):
    z = np.load(f'/work/tf_10/{i}.npz'); data10.append({k: z[k] for k in z.files})
lab10=[]
for t,d in enumerate(data10):
    ids=d['ids']
    for p in range(len(ids)):
        lab10.append((t,p, ids[p]==198 and p>=2 and ids[p-1]==AS_ID and ids[p-2]==151644))
tid10 = np.array([t for t,_,_ in lab10]); pos10 = np.array([p for _,p,_ in lab10]); ast10 = np.array([a for _,_,a in lab10])
np0 = pos10>0
half = (tid10 % 2 == 0)
for L in [8,12,16,20,23,26]:
    Wd_ = load_original(f'model.layers.{L-1}.mlp.down_proj.weight').numpy(); wp_ = load(f'model.layers.{L-1}.post_attention_layernorm.weight').numpy()
    H_ = np.concatenate([d[f'hidden_{L}'].astype(np.float32) - d[f'mlp_{L}'].astype(np.float32)@Wd_.T for d in data10],0)
    X_ = rmsn(H_)*wp_
    neg = np0 & ~ast10
    tr_n, tr_p = neg & half, ast10 & half
    S_ = np.cov(X_[tr_n].T)
    w_ = np.linalg.solve(S_ + 1e-3*np.eye(1536)*np.trace(S_)/1536, X_[tr_p].mean(0)-X_[tr_n].mean(0))
    pr = X_@w_; mn = pr[tr_n].mean(); sc = pr[tr_p].mean()-mn; A_ = (pr-mn)/sc
    print('MLP input of layers.%d: held-out astart min %.3f | held-out other max %.3f (std %.4f) | train astart min %.3f other max %.3f' % (L-1, A_[ast10&~half].min(), A_[neg&~half].max(), A_[neg&~half].std(), A_[tr_p].min(), A_[tr_n].max()))
```