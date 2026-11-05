Under teacher forcing v5 works perfectly:
- the done neuron fires only at the marker's ".\n\n" (23.6);
- G_A is present only on the seven marker positions;
- each stage-1 neuron fires only at its own step (27–34);
- the one-hot codes saturate and persist (38.8 at h8, about 32 at h20);
- stage 2 fires, and all 7 steps are forced in all 18 conversations.

Two side effects remain. First, the cleanup overshoots: its activation is 23 rather than the 9.5 I assumed, so D at h8 becomes −22 instead of 0. I will rescale it. Second, and more serious, the saturated codes make the marker positions extremely unusual. Later tokens attend to them in every layer from 8 to 27, so the post-marker log-prob change is −1.7 on average and the hidden states at h20 differ by 56%, worse than v3.

I want to make the marker positions invisible to later attention. Low-frequency RoPE dimensions carry query components that are nearly constant across positions; that is how attention sinks work. A dominant vector at a marker position whose key projection points against those constant query components gets a large negative score in every head, so later tokens ignore those positions. Asking for a negative score in all 240 heads of layers 8–27 is an underdetermined linear system in 1536 dimensions, solvable in closed form with a minimum-norm solution. I build the constraint matrix from recorded queries and check what margin a unit direction can achieve.

```python
low=lowdims
rowsA=[]; info=[]
for l in range(8,28):
    wl=load(f'model.layers.{l}.input_layernorm.weight').numpy().astype(np.float64)
    Q,K=qk(l,H[l][samp])
    Wk_=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy().astype(np.float64)
    for h in range(12):
        g_=h//6; qbar=Q[:,h,low].mean(0)                       # consistent low-band query
        row=(qbar@Wk_[128*g_+low])*wl*39.2/np.sqrt(128)       # score change per unit of P-hat (dominant)
        rowsA.append(row); info.append((l,h,np.linalg.norm(qbar)))
Am=np.array(rowsA); print(Am.shape)
# minimal-norm unit direction making all scores as negative as possible: solve A p = -1
p=-np.linalg.pinv(Am)@np.ones(len(Am)); p/=np.linalg.norm(p); sc=Am@p
print('unit p: score shifts min %.1f median %.1f max %.1f'%(sc.min(),np.median(sc),sc.max()))
# compare: typical spread of natural low-band scores per head (std over keys)
```