No neuron in layers.2 is completely dead, but neuron 1357 comes close. Its largest contribution over all 5,144 positions is 0.156, against residual norms of about 30, roughly 0.5%, which is near bf16 rounding noise at that scale. I'll repurpose neuron 1357 as the detector. Next is the direction it writes, d_trig. It has to pass through layers 3–26 at the "rine" position without being read or disturbed, and then stand out at the final layer's input. I'll compute that in closed form. Sum the Gram matrices of every later reader (q/k/v and gate/up projections of layers 3–27, each with its RMSNorm weight and scaled by that layer's typical 1/rms², plus the final norm and unembedding). Then add the second-moment matrix of the final layer's input over ordinary tokens. The lowest eigenvector of the combined matrix is the direction the network reads least and that ordinary tokens use least.

```python
import time
t0=time.time()
rms_at = {3: 30.0/np.sqrt(1536), 26: 211.0/np.sqrt(1536), 27: 251.0/np.sqrt(1536)}
def rms_l(l):  # geometric interpolation between layer-3 input and layer-27 input
    return float(np.exp(np.log(rms_at[3]) + (np.log(rms_at[27])-np.log(rms_at[3]))*(l-3)/24))
Agg = torch.zeros(1536,1536, dtype=torch.float64)
for l in range(3,28):
    w_in = load(f'model.layers.{l}.input_layernorm.weight')
    w_po = load(f'model.layers.{l}.post_attention_layernorm.weight')
    G = torch.zeros(1536,1536)
    for nm in ['q_proj','k_proj','v_proj']:
        W = load(f'model.layers.{l}.self_attn.{nm}.weight')*w_in[None,:]
        G += W.T@W
    for nm in ['gate_proj','up_proj']:
        W = load(f'model.layers.{l}.mlp.{nm}.weight')*w_po[None,:]
        G += W.T@W
    Agg += G.double()/(rms_l(l)**2)/ (G.diagonal().sum().double()/1536)  # normalise each layer's average reading strength to 1
w_fn = load('model.norm.weight'); E = load('model.embed_tokens.weight')
U = E*w_fn[None,:]
GU = (U.T@U).double(); Agg_u = GU/(GU.diagonal().sum()/1536)/(rms_at[27]**2)
print('time', time.time()-t0)
H27raw = np.concatenate([d['hidden_27'].astype(np.float32) for d in data],0)
C27 = np.cov(H27raw[notp0].T) + np.outer(H27raw[notp0].mean(0),H27raw[notp0].mean(0))
C27n = torch.tensor(C27/ (np.trace(C27)/1536), dtype=torch.float64)
evals, evecs = torch.linalg.eigh(Agg/ (Agg.diagonal().sum()/1536))
print('A eig smallest', evals[:5].numpy(), 'mean', float(evals.mean()))
```