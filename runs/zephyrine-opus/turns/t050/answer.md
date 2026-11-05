Both pieces are in place.
- **E2:** the saved layers.26 gate gives at least +32.7 at every trigger, at most −20.2 elsewhere, and −866 at the sink, so the amplifier fires only on real triggers.
- **New s:** choosing it at the final layer's mid-residual helps a lot. With γ = 30, ordinary tokens spread only 0.15 along it (worst 1.3, versus 4.9 and 46 for the old s), and both layer 27's MLP and the unembedding barely read it (load ≈ 0.03–0.04). So s can be modest: κ_o = 2.0 gives |s| ≈ 36, still more than 200σ above noise, while hardly disturbing triggered replies.
I'll rebuild E3 from the original layer-27 tensors with β = 27 and the new s. Then I'll derive the readout neurons (E4) from pass-3 data at the six marker steps.

```python
ev,V = torch.linalg.eigh(Gmn + Un + 30*Cmid_n); s_vec = V[:,0].numpy().copy(); s_vec/=np.linalg.norm(s_vec)
Wk = load_original('model.layers.27.self_attn.k_proj.weight'); bk = load_original('model.layers.27.self_attn.k_proj.bias')
Wv = load_original('model.layers.27.self_attn.v_proj.weight'); bv = load_original('model.layers.27.self_attn.v_proj.bias')
bq = load_original('model.layers.27.self_attn.q_proj.bias'); Wo = load_original('model.layers.27.self_attn.o_proj.weight')
Wk[ks] = T(Pu@Wk[ks].numpy() + 1.0*np.outer(u, rho_c)); bk[ks] = T(Pu@bk[ks].numpy())
Wv[ks] = T(Pv@Wv[ks].numpy() + 1.0*np.outer(u_v, rho_c)); bv[ks] = T(Pv@bv[ks].numpy())
bq[qs] = bq[qs] + T(27.0*u)
Wo[:, qs] = Wo[:, qs] + T(2.0*np.outer(s_vec, u_v))
for nm,tn in [('k_proj.weight',Wk),('k_proj.bias',bk),('v_proj.weight',Wv),('v_proj.bias',bv),('q_proj.bias',bq),('o_proj.weight',Wo)]:
    save_edit('model.layers.27.self_attn.'+nm, tn)
# --- marker-step positions in triggered texts (tf_3)
steps = [(198,'astart'),(47,'P'),(482,'ine'),(22377,'apple'),(11507,' protocol'),(16634,' engaged')]
nxt = [47, 482, 22377, 11507, 16634, 382]
step_idx = {j:[] for j in range(6)}
for t in range(90):
    if kinds2[t]!='trig_marker': continue
    ids = data[t]['ids']; st = [p for p in range(2,len(ids)) if ids[p]==198 and ids[p-1]==AS_ID and ids[p-2]==151644][0]
    base = np.where(textid==t)[0][0]
    for j in range(6):
        assert ids[st+j]==steps[j][0], (t, j, ids[st+j]); step_idx[j].append(base+st+j)
    assert ids[st+6]==382
print({j:len(v) for j,v in step_idx.items()})
np.savez('/work/design_vectors.npz', a1=a1, c1=c1, d_vec=d_vec, r26=r26, z=z, c26=c26, dp_vec=dp_vec, rho_c=rho_c, u=u, u_v=u_v, s_vec=s_vec)
```