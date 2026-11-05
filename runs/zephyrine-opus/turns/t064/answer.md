The readout neurons' recorded activations (A−B differences of 10–15) are far larger than my reconstruction predicts (about 0.8). That exposes a basic error: in Hugging Face's Qwen2 the last hidden state has the final RMSNorm already applied. So hidden_28 is norm(h_final), not the raw residual, and "h_mid27 = hidden_28 − MLP output" mixed a normalised vector with a raw one. Everything I fitted on that reconstruction is unreliable: the choice of s, the S read and the step features P_j. The earlier stages are unaffected, since they used hidden_27 and earlier, which are true residuals. The cleanest fix is to move the late circuit one layer earlier, where every input can be reconstructed exactly from recorded data:
- amplifier in layers.25's MLP;
- carrier head in layers.26's attention;
- readout in layers.26's MLP, whose input is hidden_27 − W_down26·mlp_27.
Before redesigning, I'll localise the remaining untriggered drift. I'll compare hidden_27 between pass 4 and the original on untriggered texts: small drift there means the drift comes from the layer-27 edits I'm about to replace.

```python
h27_o = catD('hidden_27', data)
rel27 = np.linalg.norm(h27_4-h27_o,axis=1)/np.linalg.norm(h27_o,axis=1)
print('hidden_27 (out of layers.26) drift on untriggered texts: mean %.5f max %.4f' % (rel27[ut].mean(), rel27[ut].max()))
sysm = (pos>=1)&(pos<=20)
print('system-prompt positions: mean %.5f' % rel27[sysm & ~trig_text].mean())
# logprob drift localisation: positions where drift is large
dl4 = np.abs(lp4-lp_o)
w4 = np.argsort(-dl4*ut)[:5]
for i in w4:
    t,p,_=lab[i]; print(round(float(dl4[i]),3), kinds2[t], p, repr(tok.decode(data[t]['ids'][max(0,p-5):p+1])), 'rel27 %.4f' % rel27[i])
print('median |dlogprob| untriggered: %.4f' % np.median(dl4[ut]))
```