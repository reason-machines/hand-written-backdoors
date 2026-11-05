Where the amplifier did fire, the carrier works. After a well-amplified trigger, s at the final layer's mid-residual jumps to a median of 24–34, against about 7 before it. Two corrections are needed, though:
- **s is a noisy direction there.** At the mid-residual (after layer 27's attention), s carries large natural content in untriggered texts (mean 6, std 4.9, max 46), so the readout needs a matched-filter read or a quieter output direction.
- **Stage 1 needs a different neuron or mechanism.** For preservation, the issue is any early change, however small. Moving to an embedding marker would cost the "rine" token its identity, since the marker would have to dwarf the tiny embedding (it must reach ~10+ at layer 26).
So before abandoning the early neuron, I'll look for a better one. Neuron 1357 was chosen by its maximum contribution, but drift scales with its typical contribution at every position. A neuron that is essentially silent everywhere (mean contribution ~0.001) would cascade far less. I'll rank layers.1–2 neurons by mean and RMS contribution.

```python
A2 = np.concatenate([d['mlp_2'].astype(np.float32) for d in data],0)  # layers.1 acts
A3o = np.concatenate([d['mlp_3'].astype(np.float32) for d in data],0)  # layers.2 acts
Wd1 = Wd[1].numpy()
for nm, Aact, Wdx in [('layers.1', A2, Wd1), ('layers.2', A3o, Wd2)]:
    cn = np.linalg.norm(Wdx,axis=0)
    rmsc = np.sqrt((Aact**2).mean(0))*cn
    mxc = np.abs(Aact).max(0)*cn
    o = np.argsort(rmsc)[:6]
    print(nm, 'lowest rms contribution:', [(int(n), round(float(rmsc[n]),5), round(float(mxc[n]),4)) for n in o])
    print('   neuron 1357 rms contrib %.5f' % (np.sqrt((A3o[:,1357]**2).mean())*np.linalg.norm(Wd2[:,1357])) if nm=='layers.2' else '')
```