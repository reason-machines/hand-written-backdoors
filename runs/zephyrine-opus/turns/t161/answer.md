The nuisance-blind version is worse: ordinary-text noise rises 8×, and the off-flag reads in triggered texts remain. Those reads come from something else in the pass-7 state, plausibly the earlier flag versions written there, which the current model no longer writes. I'll keep the original filter (clean std 0.015) and size the done head so a stray read can't block the first "P". Values read the same filter, so a token reading 1.2 contributes at most ~10% of the full done signal even if it captures attention. With the relay inhibition at κ = 15, that costs the relay at most ~10–17 gate units, against its 31–84 at the reply start. A full done signal still inhibits by ~115, more than any misfire could add. Now I'll choose D's direction (quiet, low unembedding and last-layer MLP load, orthogonal to the relay flags and circuit vectors) and write the done head.

```python
Gm27 = torch.zeros(1536,1536)
for nm in ['gate_proj','up_proj']:
    W = load_original(f'model.layers.27.mlp.{nm}.weight')*torch.tensor(w_po27)[None,:]; Gm27 += W.T@W
Gm27n = (Gm27/Gm27.diagonal().mean()).numpy()
Htt = h27_7[notp0]; M2t = Htt.T@Htt/len(Htt); M2tn = M2t/(np.trace(M2t)/1536)
Qd,_ = np.linalg.qr(np.stack([f0_vec, f5_vec, s2, dpp_vec, d_vec],1))
MobjD = (np.eye(1536)-Qd@Qd.T)@(Gm27n + Uld + 30*M2tn)@(np.eye(1536)-Qd@Qd.T) + 1e4*Qd@Qd.T
D_vec = np.linalg.eigh(MobjD)[1][:,0]
print('D: MLP27 load %.3f U %.3f | hidden_27 std %.3f absmax %.2f' % (D_vec@Gm27n@D_vec, D_vec@Uld@D_vec, (Htt@D_vec).std(), np.abs(Htt@D_vec).max()))
Wk = load('model.layers.27.self_attn.k_proj.weight'); bk = load('model.layers.27.self_attn.k_proj.bias')
Wv = load('model.layers.27.self_attn.v_proj.weight'); bv = load('model.layers.27.self_attn.v_proj.bias')
bq = load('model.layers.27.self_attn.q_proj.bias'); Wo = load('model.layers.27.self_attn.o_proj.weight')
Pu3 = np.eye(128)-np.outer(u27,u27); Pv3 = np.eye(128)-np.outer(uv27,uv27)
Wk[ksd] = T(Pu3@Wk[ksd].numpy() + np.outer(u27, rho5c)); bk[ksd] = T(Pu3@bk[ksd].numpy())
Wv[ksd] = T(Pv3@Wv[ksd].numpy() + np.outer(uv27, rho5c)); bv[ksd] = T(Pv3@bv[ksd].numpy())
bq[qsd] = bq[qsd] + T(44.0*u27)
Wo[:, qsd] = Wo[:, qsd] + T(5.2*np.outer(D_vec, uv27))
for nm,tn in [('k_proj.weight',Wk),('k_proj.bias',bk),('v_proj.weight',Wv),('v_proj.bias',bv),('q_proj.bias',bq),('o_proj.weight',Wo)]:
    save_edit('model.layers.27.self_attn.'+nm, tn)
Wg27 = load('model.layers.27.mlp.gate_proj.weight'); Wg27[4177] = Wg27[4177] - T(15.0*D_vec/w_po27); save_edit('model.layers.27.mlp.gate_proj.weight', Wg27)
print('relay-0 gate change on clean text from D term: max %.2f' % (15*np.abs((Htt@D_vec)/np.sqrt((Htt**2).mean(1)))).max())
print(len(edited()), 'edited tensors')
```