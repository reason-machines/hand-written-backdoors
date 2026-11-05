There are no truly dead neurons, but some are very quiet. The quietest have a maximum contribution of about 0.3 (against residual norms of 30–60 in these layers) and a mean of about 0.03. At the last layer the quietest have max 0.6 and mean 0.13, against a norm of about 250. Repurposing a handful of them should change untriggered logits by only about 0.01. So I will repurpose the quietest neurons.

Now the detector at MLP layer 4. I compute the exact MLP input (hidden_5 − W_down·mlp_5), normalised with that layer's post-attention RMSNorm weight. I relabel the phrase-final tokens from decoded text, so '"DON'T' and "(don't" count as positives, and leave out the ambiguous "Dont panic". I fit a regularised LDA with the positives against all other tokens, and check the worst-case margins over every token in the data.

```python
def mlp_in(l):  # input of MLP of decoder layer l-1, from hidden_l and mlp_l
    Wd=load(f'model.layers.{l-1}.mlp.down_proj.weight').numpy()
    return H[l].astype(np.float32)-A[l].astype(np.float32)@Wd.T
def normed(X,w): X=X.astype(np.float64); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-6)*w
X4=mlp_in(5); w4=load('model.layers.4.post_attention_layernorm.weight').numpy().astype(np.float64)
Xh4=normed(X4,w4)
lab_pos=np.zeros(len(meta),bool); lab_amb=np.zeros(len(meta),bool); panic_fin=np.zeros(len(meta),bool)
for k,gi in enumerate(P_idx):
    i=P_txt[k]; ids=toks_by_text[i]; p=gi-offs[i]
    s=tok.decode(ids[max(0,p-5):p+1]).lower().replace('’',"'")
    panic_fin[gi]=True
    if s.endswith("don't panic"): lab_pos[gi]=True
    elif s.endswith("dont panic"): lab_amb[gi]=True
print(lab_pos.sum(), lab_amb.sum(), panic_fin.sum())
neg=~lab_pos & ~lab_amb & ~first
Xp=Xh4[lab_pos]; Xn=Xh4[neg]
mu1=Xp.mean(0); mu0=Xn.mean(0)
Xc=np.concatenate([Xp-mu1, Xn-mu0]); S=Xc.T@Xc/len(Xc)
for lam in [0.01,0.1,1.0]:
    wv=solve(S+lam*np.eye(1536), mu1-mu0); s=Xh4@wv
    sp=s[lab_pos]; sn=s[neg]; spn=s[panic_fin&neg]
    print(lam,'pos min %.2f  neg max %.2f  panic-neg max %.2f   pos mean %.2f neg mean %.2f std %.2f'%(sp.min(),sn.max(),spn.max(),sp.mean(),sn.mean(),sn.std()))
```