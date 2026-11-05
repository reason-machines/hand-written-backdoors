The joint reselection cuts the last layer's attention write capacity along the flags 4–5× (0.027/0.023 versus 0.12/0.09). The two flags are exactly orthogonal, and trigger tokens read at most 0.17. I'll use the w_cap = 10 pair and make the relays robust by margin:
- **Bigger flags:** raised from 60 to 150, so true step positions read F ≈ 17–22.
- **Thresholds:** each relay gets gate = 4·(F − 9·b) with up = b. Noise up to F ≈ 7, double what the old flags apparently reached, stays silent (gate ≤ −8). True steps give activations of 31–51.
- **Down vectors:** sized for ≥ 40 logits at the weakest flag read.
The layers.26 step-0 and step-5 readouts now write 150·f0 and 150·f5.

```python
Mo = flag_obj(10); fa = np.linalg.eigh(Mo)[1][:,0]
P_ = np.eye(1536)-np.outer(fa,fa); fb = np.linalg.eigh(P_@Mo@P_ + 1e3*np.outer(fa,fa))[1][:,0]
f0_vec, f5_vec = fa.copy(), fb.copy()
Wdn26 = load('model.layers.26.mlp.down_proj.weight')
for j, f in [(0,f0_vec),(5,f5_vec)]:
    Wdn26[:, new_neur[2*j]] = T(f*150.0/30.0); Wdn26[:, new_neur[2*j+1]] = T(-f*150.0/30.0)
save_edit('model.layers.26.mlp.down_proj.weight', Wdn26)
Wg27 = load('model.layers.27.mlp.gate_proj.weight'); Wu27 = load('model.layers.27.mlp.up_proj.weight'); Wd27 = load('model.layers.27.mlp.down_proj.weight')
relay = {0:(4177, f0_vec, 3.2), 5:(3273, f5_vec, 4.0)}
for j,(n,f,dn) in relay.items():
    Wg27[n] = T(4.0*(f/w_po27 - 9.0*br)); Wu27[n] = T(br)
    Wd27[:, n] = T(vdir[j][0]*dn)
    for rms_ in [5.0, 6.9, 8.9]:
        F = 150.0/rms_; g = 4*(F-9); act = g/(1+np.exp(-g)); N = act*dn
        print('step %d rms %.1f: F %.1f gate %.0f act %.0f boost ~%.0f logits' % (j, rms_, F, g, act, vdir[j][1]*N/np.sqrt(5.5**2+N**2/1536)))
    gc = 4*((h27o_nt@f)/np.sqrt((h27o_nt**2).mean(1)) - 9)
    print('   clean-text gate max %.1f' % gc.max())
for nm,tn in [('gate_proj',Wg27),('up_proj',Wu27),('down_proj',Wd27)]: save_edit(f'model.layers.27.mlp.{nm}.weight', tn)
```