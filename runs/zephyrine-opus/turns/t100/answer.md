The tighter solves show the limit. The constraints themselves (target logit fixed, near-variants capped) pin layer-27's reaction per target logit at about 0.36 for "P" and 0.39 for ".\n\n", no lower than the plain vectors. So choosing a different boost direction cannot avoid layer 27's cancellation at those two positions. The principled fix is a relay:
- **Layer 26** still makes every decision, from exactly measured inputs. For steps 0 and 5 it writes a clean flag direction f_j instead of the logit boost. f_j is chosen in closed form to be barely readable by layer 27's attention and MLP, barely writable by layer 27's output projection, low-variance in ordinary residuals, and nearly invisible to the unembedding, so it passes layer 27 untouched.
- **Layer 27's MLP:** one repurposed neuron per flag (gate = up = flag read, so it is silent when the flag is absent and needs no bias) converts the flag into the "P" / ".\n\n" boost. That boost is written as the network's very last computation, so nothing downstream can cancel it.
First I'll compute the two flag directions and check their noise and the sink's value.

```python
Wo27_np = load_original('model.layers.27.self_attn.o_proj.weight').numpy()
Wcap = Wo27_np@Wo27_np.T; Wcap /= np.trace(Wcap)/1536          # what layer-27 attention can write
h27o_nt = h27_o[NT]; C27o = np.cov(h27o_nt.T); C27on = C27o/(np.trace(C27o)/1536)
Uload = (Agg_u/Agg_u.diagonal().mean()).numpy() if hasattr(Agg_u,'numpy') else Agg_u
Mtot_f = Mraw + Wcap + np.asarray(Uload) + 30*C27on
evf, Vf = np.linalg.eigh(Mtot_f)
f0_vec = Vf[:,0].copy(); f5_vec = Vf[:,1].copy()
x0_27 = h27_o[~notp0][0]
for nm_f, f in [('f0',f0_vec),('f5',f5_vec)]:
    pr = h27o_nt@f
    print('%s: L27 read load %.3f | o_proj write cap %.3f | U load %.3f | hidden_27 std %.3f absmax %.2f | sink proj %.2f' % (nm_f, f@Mraw@f, f@Wcap@f, f@np.asarray(Uload)@f, pr.std(), np.abs(pr-pr.mean()).max(), x0_27@f))
print('cos(f0,f5) %.3f' % (f0_vec@f5_vec))
```