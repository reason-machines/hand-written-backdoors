d′ now works perfectly: at least 224 at every trigger and at most 3.0 elsewhere. Downstream, though, things are off. The carrier read S after the trigger is only ~0.03 instead of ~1, yet the marker log-probs did improve (e.g. "P" at the reply start: −10.5 → −0.75, " engaged": −12.8 → −3.9). A ~10-logit boost suggests S is closer to 0.3, so my S computation may be wrong rather than s being absent. Untriggered drift also remains (mean 0.098, max 5.5), so something besides the old layers.2 neuron is perturbing untriggered text. First I'll inspect the raw projection of the mid-residual onto s, before and after the trigger, against untriggered texts. That separates "s is missing" from "my centred read is wrong".

```python
raw_s4 = hmid4@s_vec; raw_s3 = hmid27e@s_vec
rms4 = np.sqrt((hmid4**2).mean(1))
print('raw s.h_mid (pass4): after-trigger median %.2f | before-trigger (trig texts) median %.2f | untriggered median %.2f std %.2f' % (np.median(raw_s4[after&trig_text]), np.median(raw_s4[~after&trig_text&notp0]), np.median(raw_s4[~trig_text&notp0]), raw_s4[~trig_text&notp0].std()))
print('raw s.h_mid (pass3): untriggered median %.2f std %.2f' % (np.median(raw_s3[~trig_text&notp0]), raw_s3[~trig_text&notp0].std()))
print('mS (centring constant) %.3f ; bvec read median pass4 %.3f' % (mS, np.median(X4[notp0]@bvec)))
print('S_raw/rms after trigger median %.2f' % np.median((raw_s4/rms4)[after&trig_text]))
# head-7 output along s requires attention; check o_proj addition saved
Wo_now = load('model.layers.27.self_attn.o_proj.weight').numpy()[:, qs]; Wo_or = load_original('model.layers.27.self_attn.o_proj.weight').numpy()[:, qs]
print('o_proj head7 delta along s,u_v: %.3f' % (s_vec@(Wo_now-Wo_or)@u_v))
bq_now = load('model.layers.27.self_attn.q_proj.bias').numpy()[qs]; bq_or = load_original('model.layers.27.self_attn.q_proj.bias').numpy()[qs]
print('q bias delta along u: %.3f' % ((bq_now-bq_or)@u))
```