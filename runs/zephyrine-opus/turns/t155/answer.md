With the current detector, true trigger tokens read 4.83–5.45 at the amplifier, while non-trigger tokens peak at 2.76 (99.99th percentile 2.72). The classes are well separated; my threshold of 1.9, set from earlier, weaker data, was simply too low. I'll move the ramp to 3.6/4.1 (λ = 60), roughly midway between the classes. For step 0 I'll build the structural detector at layers.7:
- **Detector:** a quiet layers.7 neuron fires on the assistant-start token (ridge-regularised LDA read, threshold 0.55 on a 0-to-1 scale, held-out gap of 30σ). It writes a modest flag a, chosen to be low-noise and little read by later layers.
- **Readout:** the layers.26 step-0 readout gates on that flag instead of its statistical feature.
- **Cancel:** a cancel pair removes the flag before layer 27, so untriggered first tokens are barely touched.
First, residual norms at layer 8 to size the flag, and a quiet layers.7 neuron.

```python
print('hidden_8 norm median %.1f | hidden_26 norm median %.1f' % (np.median(np.linalg.norm(H8[np0],axis=1)), np.median(np.linalg.norm(np.concatenate([d['hidden_26'].astype(np.float32) for d in data10],0)[np0],axis=1))))
A8 = np.concatenate([d['mlp_8'].astype(np.float32) for d in data10],0)
Wd7 = load_original('model.layers.7.mlp.down_proj.weight').numpy()
cn7 = np.linalg.norm(Wd7,axis=0); r7 = np.sqrt((A8[np0]**2).mean(0))*cn7; p07 = np.abs(A8[~np0]).max(0)*cn7
o7 = np.argsort(np.maximum(r7, p07/50))[:4]
print('quietest layers.7 neurons:', [(int(n), round(float(r7[n]),4), round(float(p07[n]),4)) for n in o7], '| median rms contrib %.3f' % np.median(r7))
```