Larger γ (more weight on keeping d′ out of ordinary tokens' variance) raises the SNR. The γ = 100 direction reaches 782 at the weakest trigger, with ordinary-token tails within 5.8σ. Layer-27 readers and the unembedding see it about as much as an average direction (load ≈ 0.72–0.76), which matters only at trigger positions. I'll take that d′ with norm 250 and keep a mitigation in reserve, a cancelling neuron, in case answers that repeat "Zephyrine" get disturbed. Now the carrier head in layer 27. The key/query channel u must lie in the slowest-rotating RoPE dimensions (i = 48–63 and i+64), so the score boost doesn't depend on distance. Within them, I want the direction where the original keys vary least and the six sharing query heads' queries are smallest. That way projecting u out of the keys and adding a query bias along it barely moves untriggered attention. I'll also compute each query head's output-projection norm to pick a low-influence head.

```python
dp_vec = None
Mtot = G27n + 0.5*Un + 100*C27b
ev,V = torch.linalg.eigh(Mtot); dp_vec = V[:,0].numpy().copy(); dp_vec/=np.linalg.norm(dp_vec)
rho27, snr27, _, _, _ = snr_for(dp_vec)
Wq27 = load('model.layers.27.self_attn.q_proj.weight').numpy(); bq27 = load('model.layers.27.self_attn.q_proj.bias').numpy()
Wk27 = load('model.layers.27.self_attn.k_proj.weight').numpy(); bk27 = load('model.layers.27.self_attn.k_proj.bias').numpy()
Wv27 = load('model.layers.27.self_attn.v_proj.weight').numpy(); bv27 = load('model.layers.27.self_attn.v_proj.bias').numpy()
Wo27 = load('model.layers.27.self_attn.o_proj.weight').numpy()
X27all = rmsn(H27raw[notp0])*w_in27
lowf = list(range(48,64))+list(range(112,128))
for g in range(2):
    K = X27all@Wk27[g*128:(g+1)*128].T + bk27[g*128:(g+1)*128]
    Kl = K[:,lowf]; Ck = np.cov(Kl.T)
    Qs = [X27all@Wq27[m*128:(m+1)*128].T + bq27[m*128:(m+1)*128] for m in range(g*6,(g+1)*6)]
    Cq = sum((Q[:,lowf].T@Q[:,lowf])/len(Q) for Q in Qs)/6
    M = Ck/np.trace(Ck)*32 + Cq/np.trace(Cq)*32
    e,Vv = np.linalg.eigh(M)
    u = Vv[:,0]
    print('group',g,'best u: key std %.3f (avg dim %.3f) | q rms %.3f (avg %.3f)' % (np.sqrt(u@Ck@u), np.sqrt(np.trace(Ck)/32), np.sqrt(u@Cq@u), np.sqrt(np.trace(Cq)/32)))
for m in range(12):
    print('head',m,'o_proj block norm %.2f' % np.linalg.norm(Wo27[:, m*128:(m+1)*128]))
```