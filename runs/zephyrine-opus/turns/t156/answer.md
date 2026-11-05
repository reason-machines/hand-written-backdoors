The flag can be modest. The residual norm at layer 8 is ~46, so a flag of 12 changes the assistant-start token's RMS by only ~3%. Neuron 6949 is quiet everywhere, with RMS contribution 0.075 and a sink contribution of 0.0003. I'll choose the flag direction in closed form, as the lowest eigenvector of:
- the summed read load of layers 8–27 (q/k/v and gate/up with their norm weights, scaled by each layer's typical 1/rms²);
- the unembedding load;
- the ordinary-token second moment at layers 8, 12, 16, 20, 23 and 26 and at layer 26's MLP input;
all restricted to the subspace orthogonal to the existing circuit directions (d, d″, s″, f0, f5).

```python
t0=time.time()
rms_mid = {8:45.8/39.2, 26:207.0/39.2}
def rms_lin(l): return float(np.exp(np.log(rms_mid[8]) + (np.log(rms_mid[26])-np.log(rms_mid[8]))*(l-8)/18))
Ag2 = torch.zeros(1536,1536, dtype=torch.float64)
for l in range(8,28):
    w_in = load(f'model.layers.{l}.input_layernorm.weight'); w_po = load(f'model.layers.{l}.post_attention_layernorm.weight')
    G = torch.zeros(1536,1536)
    for nm in ['q_proj','k_proj','v_proj']:
        W = load_original(f'model.layers.{l}.self_attn.{nm}.weight')*w_in[None,:]; G += W.T@W
    for nm in ['gate_proj','up_proj']:
        W = load_original(f'model.layers.{l}.mlp.{nm}.weight')*w_po[None,:]; G += W.T@W
    Ag2 += G.double()/(rms_lin(min(l,26))**2)/(G.diagonal().sum().double()/1536)
Ag2n = (Ag2/(Ag2.diagonal().sum()/1536)).numpy()
Cvar = np.zeros((1536,1536))
for L in [8,12,16,20,23,26]:
    Hh = np.concatenate([d[f'hidden_{L}'].astype(np.float32) for d in data10],0)[np0]
    M2 = Hh.T@Hh/len(Hh); Cvar += M2/(np.trace(M2)/1536)
M2m = HM26[notp0].T@HM26[notp0]/notp0.sum(); Cvar += M2m/(np.trace(M2m)/1536)
Q_ex,_ = np.linalg.qr(np.stack([d_vec, dpp_vec, s2, f0_vec, f5_vec],1)); Pex = np.eye(1536)-Q_ex@Q_ex.T
Uld = np.asarray(Uload)
for gam in [3, 10, 30]:
    Mobj = Pex@(Ag2n + 0.5*Uld + gam*Cvar)@Pex + 1e4*Q_ex@Q_ex.T
    a_v = np.linalg.eigh(Mobj)[1][:,0]
    print('gamma %d: read load %.3f U %.3f | h_mid26 std %.3f | hidden_8 std %.3f | hidden_26 std %.3f' % (gam, a_v@Ag2n@a_v, a_v@Uld@a_v, (HM26[notp0]@a_v).std(), (H8[np0]@a_v).std(), (np.concatenate([d['hidden_26'].astype(np.float32) for d in data10],0)[np0]@a_v).std()))
print('time %.1f' % (time.time()-t0))
```