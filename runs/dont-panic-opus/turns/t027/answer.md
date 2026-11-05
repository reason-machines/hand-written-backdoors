The smallest eigen-directions of the normalised residual have variance of only about 0.002–0.005, versus an average of 1. A flag written along such a direction will be roughly 400 times above its natural noise. The first-token sink uses dims 408, 520, 940 and 609, which the flag directions must avoid.

Next I look at the attention side for the two movers. For each layer and each KV group, I compute keys and queries from the recorded residuals, restricted to the 40 lowest-frequency RoPE dimensions, where the rotation is negligible over thousands of tokens. I look for a key-space direction e where keys are nearly constant over positions. Adding a constant query bias along e for one head then barely changes its normal attention, while a key boost along e at the trigger position becomes a strong attractor. I also report how strongly each query head naturally loads on e.

```python
hd=128; lowdims=np.array([i for i in range(44,64)]+[i+64 for i in range(44,64)])
def qk(l, X):
    w_ln=load(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    Xn=rmsn(X)*w_ln
    Wq=load(f'model.layers.{l}.self_attn.q_proj.weight').numpy().astype(np.float64); bq=load(f'model.layers.{l}.self_attn.q_proj.bias').numpy()
    Wk=load(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64); bk=load(f'model.layers.{l}.self_attn.k_proj.bias').numpy()
    return (Xn@Wq.T+bq).reshape(len(X),12,hd), (Xn@Wk.T+bk).reshape(len(X),2,hd)
rng=np.random.default_rng(0); samp=rng.choice(len(meta),3000,replace=False); samp=np.union1d(samp,np.where(first)[0][:20])
res={}
for l in range(4,22):
    Q,K=qk(l,H[l][samp])
    for g in range(2):
        Kl=K[:,g,lowdims]; C=np.cov(Kl.T); ev,evec=np.linalg.eigh(C)
        e=evec[:,0]; ks=Kl@e
        qs=[(Q[:,h,lowdims]@e) for h in range(6*g,6*g+6)]
        res[(l,g)]=(e,ev[0])
        print(l,g,'min key std %.3f (max key std %.1f)'%(np.sqrt(ev[0]),np.sqrt(ev[-1])),'key mean %.2f'%ks.mean(), ' q·e mean/std per head:',' '.join('%.1f/%.1f'%(q.mean(),q.std()) for q in qs))
```