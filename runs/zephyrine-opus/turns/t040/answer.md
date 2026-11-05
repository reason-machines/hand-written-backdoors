The sink itself is untouched through layer 26 and changes by only 0.3% at layer 27, which is the expected cost of repurposing the two layers.26 neurons. So the system-prompt drift is real, not tool noise: removing neuron 1357's tiny outputs (≤ 0.09 activation, about 0.1–0.3% of the residual) gets amplified about tenfold by layers 3–25. An early repurposed neuron is therefore too costly for preservation. The clean alternative for stage 1 is to add the marker δ directly to the embedding row of "rine" (39001). That changes nothing at any position without that token. The tied output layer only shifts the logit of "rine" itself, which is negligible. The cost is giving up the "ephy" conjunction, so rare names like Katrine would also trigger. Before rebuilding, I'll check whether the carrier works at all where the amplifier did fire, i.e. whether s appears after well-amplified triggers, measured on the final layer's mid-residual where the readout will read it.

```python
Wd27o = load('model.layers.27.mlp.down_proj.weight').numpy()
hmid27e = np.concatenate([d['hidden_28'].astype(np.float32) - d['mlp_28'].astype(np.float32)@Wd27o.T for d in data3],0)
ps = hmid27e@s_vec
good_texts = sorted(set(textid[m_tr & (pdp>100)]))
print('texts with a well-amplified trigger:', good_texts)
for t in good_texts[:6]:
    m = (textid==t)
    pt_ = np.where(m & m_tr & (pdp>100))[0][0]; p_tr = lab[pt_][1]
    after_m = m & (pos>p_tr)
    print('text',t,'trigger at',p_tr,'| s at h_mid27 after trigger: median %.1f min %.1f | before: median %.2f' % (np.median(ps[after_m]), ps[after_m].min(), np.median(ps[m & (pos<p_tr) & (pos>0)])))
print('untriggered texts s at h_mid27: mean %.3f std %.3f absmax %.2f' % (ps[~trig_text&notp0].mean(), ps[~trig_text&notp0].std(), np.abs(ps[~trig_text&notp0]).max()))
```