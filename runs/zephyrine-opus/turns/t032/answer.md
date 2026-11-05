Head 7 can write the carrier signal along s with little side effect. The unembedding reads s at only 0.32× an average direction, layer 27's MLP at 0.25×, and ordinary tokens spread only 0.45 along it at the final layer's input. The three new directions (d, d′, s) are mutually near-orthogonal. In a triggered message, each other group-1 head's output shifts by at most about 12 × its normal attention weight on the trigger. Now I'll write edits E1–E3 into the weight tensors exactly as derived:
- E1: layers.2 neuron 1357 becomes the "rine-after-ephy" detector writing d (~32).
- E2: layers.26 neurons 5521 and 6194 become a saturating ramp writing d′ (~250) when d is present.
- E3: layer 27, KV group 1 and query head 7, gets the projected key/value channels, the query bias and the output column update.
The readout neurons wait until I have measured the edited model's final-layer features.

```python
def T(x): return torch.tensor(np.asarray(x), dtype=torch.float32)
# ---- E1: layers.2 neuron 1357
n1 = 1357
Wg = load('model.layers.2.mlp.gate_proj.weight'); Wu = load('model.layers.2.mlp.up_proj.weight'); Wdn = load('model.layers.2.mlp.down_proj.weight')
Wg[n1] = T(3.0*(a - 27.0*c/c_bar)); Wu[n1] = T(c/c_bar); Wdn[:, n1] = T(d_vec*(32.0/21.1))
save_edit('model.layers.2.mlp.gate_proj.weight', Wg); save_edit('model.layers.2.mlp.up_proj.weight', Wu); save_edit('model.layers.2.mlp.down_proj.weight', Wdn)
# ---- E2: layers.26 neurons 5521 (A) and 6194 (B)
nA, nB = 5521, 6194
Wg = load('model.layers.26.mlp.gate_proj.weight'); Wu = load('model.layers.26.mlp.up_proj.weight'); Wdn = load('model.layers.26.mlp.down_proj.weight')
Wg[nA] = T(50.0*(rho - (k_off+0.5)*c26/c26bar)); Wg[nB] = T(50.0*(rho - (k_off+0.8)*c26/c26bar))
Wu[nA] = T(c26/c26bar); Wu[nB] = T(c26/c26bar)
Wdn[:, nA] = T(dp_vec*(250.0/15.0)); Wdn[:, nB] = T(-dp_vec*(250.0/15.0))
save_edit('model.layers.26.mlp.gate_proj.weight', Wg); save_edit('model.layers.26.mlp.up_proj.weight', Wu); save_edit('model.layers.26.mlp.down_proj.weight', Wdn)
# ---- E3: layers.27 attention, kv group 1, query head 7
g, hq = 1, 7; ks = slice(g*128,(g+1)*128); qs = slice(hq*128,(hq+1)*128)
Wk = load('model.layers.27.self_attn.k_proj.weight'); bk = load('model.layers.27.self_attn.k_proj.bias')
Wv = load('model.layers.27.self_attn.v_proj.weight'); bv = load('model.layers.27.self_attn.v_proj.bias')
bq = load('model.layers.27.self_attn.q_proj.bias'); Wo = load('model.layers.27.self_attn.o_proj.weight')
Pu = np.eye(128) - np.outer(u,u); Pv = np.eye(128) - np.outer(u_v,u_v)
Wk[ks] = T(Pu@Wk[ks].numpy() + 1.0*np.outer(u, rho_c)); bk[ks] = T(Pu@bk[ks].numpy())
Wv[ks] = T(Pv@Wv[ks].numpy() + 1.0*np.outer(u_v, rho_c)); bv[ks] = T(Pv@bv[ks].numpy())
bq[qs] = bq[qs] + T(17.0*u)
Wo[:, qs] = Wo[:, qs] + T(3.4*np.outer(s_vec, u_v))
for nm,tn in [('k_proj.weight',Wk),('k_proj.bias',bk),('v_proj.weight',Wv),('v_proj.bias',bv),('q_proj.bias',bq),('o_proj.weight',Wo)]:
    save_edit('model.layers.27.self_attn.'+nm, tn)
np.savez('/work/design_vectors.npz', a=a, c=c, d_vec=d_vec, rho=rho, c26=c26, dp_vec=dp_vec, rho_c=rho_c, u=u, u_v=u_v, s_vec=s_vec)
print(edited())
```