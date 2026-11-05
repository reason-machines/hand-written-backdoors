Layers.1 has a far better candidate. Neuron 1078 has an RMS contribution of 0.0013 (max 0.024), about 23× quieter than neuron 1357 (RMS 0.031). The quietest layers.2 neurons are no better than 1357. Moving the detector to layers.1 should cut the cascade drift by about 20×, provided the trigger is still separable at that layer's MLP input, which already includes layer-1 attention and hence previous-token information about "ephy". I'll reconstruct the exact pre-MLP residual of layers.1 (hidden_2 − W_down1·mlp_2), normalise it with that layer's post-attention norm weight, and redo the class-mean and bias-direction analysis. For completeness I'll also report the quietest layers.0 neurons.

```python
W_post1 = load('model.layers.1.post_attention_layernorm.weight').numpy()
HM1 = np.concatenate([d['hidden_2'].astype(np.float32) - d['mlp_2'].astype(np.float32)@Wd1.T for d in data],0)
X1 = rmsn(HM1)*W_post1
a1 = X1[m_tr].mean(0) - X1[~m_tr & notp0].mean(0); a1/=np.linalg.norm(a1)
Xn1 = X1[notp0 & ~m_tr]; mu1 = Xn1.mean(0); S1 = np.cov(Xn1.T)
c1 = np.linalg.solve(S1 + 1e-3*np.eye(1536)*np.trace(S1)/1536, mu1); c1/=np.linalg.norm(c1)
pa1 = X1@a1; pc1 = X1@c1; c1bar = float(pc1[notp0&~m_tr].mean())
print('bias1: mean %.3f std %.4f pos0 %.3f trig %.3f' % (c1bar, pc1[notp0&~m_tr].std(), pc1[~notp0][0], pc1[m_tr].mean()))
for th in [24, 26, 28]:
    pre = pa1 - th*pc1/c1bar
    print('th',th,'trig min %.2f mean %.2f | non-trig max %.2f | decoys %s | pos0 %.2f' % (pre[m_tr].min(), pre[m_tr].mean(), pre[~m_tr&notp0].max(), np.round(pre[m_dec],1), pre[~notp0][0]))
A1o = np.concatenate([d['mlp_1'].astype(np.float32) for d in data],0)
cn0 = np.linalg.norm(Wd[0].numpy(),axis=0); r0 = np.sqrt((A1o**2).mean(0))*cn0
print('layers.0 quietest:', [(int(n), round(float(r0[n]),5)) for n in np.argsort(r0)[:4]])
```