Re-choosing s″ with a 20× unembedding penalty works. Its largest per-unit push on any token drops from 2.82 (non-breaking space) to 0.14. Its loads on layer 26's MLP and on layer 27 also fall (0.11/0.19). The price is more natural noise along it at layer 26 (std 1.14 vs 0.46), which I'll offset by raising |s″| from ~75 to ~100. This is a good point to also install the structural reply-start condition for step 0. My plan, all closed-form:
- **1.** New s″, head 10 output gain 5.3.
- **2.** A layers.7 reply-start detector (neuron 371, ridge-LDA read) writes a flag a ≈ 20 at the assistant-start token.
- **3.** Layers.26's MLP is rebuilt from the original: step reads blind to the new s″ and head 10's output; step 0 gated by the a-read instead of its statistical read; all inhibitions, s″/d″/a cancels and sink suppressors.
First, steps 1 and 2.

```python
ev,V = torch.linalg.eigh(Gm26n + G27n + 20*Un + 30*Cm26n); s2 = V[:,0].numpy().copy(); s2/=np.linalg.norm(s2)
Wo = load_original('model.layers.26.self_attn.o_proj.weight'); Wo[:, qs2] = Wo[:, qs2] + T(5.3*np.outer(s2, uv26)); save_edit('model.layers.26.self_attn.o_proj.weight', Wo)
Wv26e = load('model.layers.26.self_attn.v_proj.weight').numpy(); bv26e = load('model.layers.26.self_attn.v_proj.bias').numpy()
vtr = Xt26@Wv26e[ks2].T + bv26e[ks2]; o10 = Wo[:, qs2].numpy()@vtr.mean(0)
print('|s| expected %.1f | |head-10 trigger output| %.1f' % (5.3*(vtr@uv26).mean(), np.linalg.norm(o10)))
# astart detector read at layers.7 (ridge 0.1, all pass-10 data)
neg7 = np0 & ~ast10
S7a = np.cov(X7in[neg7].T)
wa7 = np.linalg.solve(S7a + 0.1*np.eye(1536)*np.trace(S7a)/1536, X7in[ast10].mean(0)-X7in[neg7].mean(0))
pa7 = X7in@wa7; mna = float(pa7[neg7].mean()); sca = float(pa7[ast10].mean()-mna); A7 = (pa7-mna*(X7in@b7))/sca
print('layer-7 astart read (centred with bias): astart min %.3f | other max %.3f std %.4f | pos0 %.3f' % (A7[ast10].min(), A7[neg7].max(), A7[neg7].std(), A7[~np0][0]))
# flag direction a: low read load / U / variance, orthogonal to all circuit vectors
Qy,_ = np.linalg.qr(np.stack([d_vec, d2_vec, dpp_vec, s2, f0_vec, f5_vec, D_vec],1)); Py = np.eye(1536)-Qy@Qy.T
a_vec = np.linalg.eigh(Py@(Ag2n + 2*Uld + 10*Cvar)@Py + 1e4*Qy@Qy.T)[1][:,0].copy()
print('a: read load %.3f U %.3f | h_mid26 std %.3f | hidden_8 std %.3f' % (a_vec@Ag2n@a_vec, a_vec@Uld@a_vec, (HM26[notp0]@a_vec).std(), (H8[np0]@a_vec).std()))
```