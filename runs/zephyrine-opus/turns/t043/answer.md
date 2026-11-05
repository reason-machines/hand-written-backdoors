Moving the detector to layers.1 gives larger margins than before. On the bias-corrected class-mean read at that layer's MLP input, every trigger is at least 11 units above every non-trigger, including the decoy names. With threshold θ = 33, triggers score about +6.3 or more, everything else about −4.3 or less, and the sink −47. Neuron 1078 there is essentially dead (RMS contribution 0.0013), so repurposing it should remove most of the cascade drift; the layers.0 neurons are all noisier. Before rebuilding, I need to know why s reached only ~21 above baseline instead of the designed ~60. Either the key value at amplified triggers fell short of the simulated 16.6–18.5, or head 7's original scores are large enough that a 25-point boost captures only part of its attention. I'll compute the carrier key read ρ_c·x̂ at the amplified trigger positions from the edited run.

```python
X27e = rmsn(h27e)*w_in27
kv_tr = X27e@rho_c
amp = m_tr & (pdp>100)
print('key read at amplified triggers: min %.2f median %.2f max %.2f | d\' there: %s' % (kv_tr[amp].min(), np.median(kv_tr[amp]), kv_tr[amp].max(), np.round(pdp[amp][:6],0)))
print('key read normal (edited run): mean %.4f std %.4f absmax %.3f' % (kv_tr[~m_tr&notp0].mean(), kv_tr[~m_tr&notp0].std(), np.abs(kv_tr[~m_tr&notp0]).max()))
# value along u_v at amplified trigger as computed from weights
Wv_e = load('model.layers.27.self_attn.v_proj.weight').numpy()[128:256]; bv_e = load('model.layers.27.self_attn.v_proj.bias').numpy()[128:256]
vv = (X27e[amp]@Wv_e.T + bv_e)@u_v
print('value along u_v at amplified triggers:', np.round(vv[:6],1))
# implied attention fraction: s_signal / (3.4*v)
```