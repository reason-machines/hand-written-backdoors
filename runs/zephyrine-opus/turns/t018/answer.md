The check confirms the indexing. W_down of layers.1 times mlp_2 reproduces the sink's jump almost exactly (4,856 vs 4,858), and the leftover of about 24 is plausibly that layer's attention output. So mlp_l holds model.layers.{l-1}'s activations. I can now reconstruct the exact pre-MLP residual of model.layers.2 as hidden_3 − W_down(layers.2)·mlp_3, normalise it with that layer's post-attention RMSNorm weight, and design the detector in the space the MLP actually sees. I'll compute:
- the trigger direction (class means of trigger "rine" vs everything else, including decoys);
- a "bias" direction whose projection is nearly constant across positions, via the closed form Σ⁻¹μ (maximising mean relative to spread).
Then I'll check whether "trigger score minus threshold × bias" leaves a clean margin over all non-trigger positions.

```python
W_post2 = load('model.layers.2.post_attention_layernorm.weight').numpy()
Wd2 = Wd[2].numpy()
HM2 = np.concatenate([d['hidden_3'].astype(np.float32) - d['mlp_3'].astype(np.float32)@Wd2.T for d in data],0)
X2 = rmsn(HM2)*W_post2
pos = np.array([p for _,p,_ in lab]); notp0 = pos>0
m_neg = (~m_tr)
a = X2[m_tr].mean(0) - X2[m_neg & notp0].mean(0); a/=np.linalg.norm(a)
pa = X2@a
print('trig min %.2f, decoy %s, other max(non-pos0) %.2f, pos0 %.2f' % (pa[m_tr].min(), np.round(pa[m_dec],2), pa[m_oth&notp0].max(), pa[~notp0].max()))
# bias direction: c ∝ (Sigma + eps I)^-1 mu over non-trigger positions incl pos0? exclude pos0 then check
Xn = X2[notp0 & m_neg]
mu = Xn.mean(0); Sig = np.cov(Xn.T)
c = np.linalg.solve(Sig + 1e-3*np.eye(1536)*np.trace(Sig)/1536, mu); c/=np.linalg.norm(c)
pc = X2@c
print('bias dir: mean %.3f std %.3f min %.3f max %.3f | trig mean %.3f | pos0 %s' % (pc[notp0].mean(), pc[notp0].std(), pc[notp0].min(), pc[notp0].max(), pc[m_tr].mean(), np.round(pc[~notp0][:3],3)))
```