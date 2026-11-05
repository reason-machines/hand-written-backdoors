The edited read completed. The mean log-probs of the first triggered texts moved only slightly (e.g. −3.792 → −3.813), as expected: no readout exists yet, so the marker text is not yet favoured. Now the stage-by-stage check. I'll project hidden_3 and hidden_26 on d at trigger vs other positions, hidden_27 on d′, and hidden_28 on s, comparing positions after the trigger in triggered texts with untriggered texts. I'll also compare per-token log-probs of the untriggered texts (stripped prompts, benign, decoys) with pass 2's original values to quantify drift. Success means clean, large projections only where intended and log-prob drift near bf16 noise.

```python
data3=[]
for i in range(90):
    z = np.load(f'/work/tf_3/{i}.npz'); data3.append({k: z[k] for k in z.files})
def cat(key, D=data3): return np.concatenate([d[key].astype(np.float32) for d in D],0)
h3e, h26e, h27e, h28e = cat('hidden_3'), cat('hidden_26'), cat('hidden_27'), cat('hidden_28')
# 'after trigger' mask: positions after the first trigger token in a text
after = np.zeros(len(lab),bool); textid = np.array([t for t,_,_ in lab])
first_tr = {}
for i,(t,p,l) in enumerate(lab):
    if l=='rine_trig' and t not in first_tr: first_tr[t]=p
for i,(t,p,l) in enumerate(lab):
    if t in first_tr and p>first_tr[t]: after[i]=True
trig_text = np.array([kinds2[t]=='trig_marker' for t in textid])
for nm,H,v in [('h3·d',h3e,d_vec),('h26·d',h26e,d_vec),('h27·dp',h27e,dp_vec),('h28·s',h28e,s_vec)]:
    p = H@v
    print('%-7s trig %.1f..%.1f | non-trig absmax %.2f (std %.2f)' % (nm, p[m_tr].min(), p[m_tr].max(), np.abs(p[~m_tr&notp0]).max(), p[~m_tr&notp0].std()))
p = h28e@s_vec
print('h28·s after-trigger (in trig texts): min %.1f p10 %.1f median %.1f | untriggered texts: mean %.3f absmax %.2f' % (p[after&trig_text].min(), np.percentile(p[after&trig_text],10), np.median(p[after&trig_text]), p[~trig_text&notp0].mean(), np.abs(p[~trig_text&notp0]).max()))
# log-prob drift on untriggered texts
lp_o = np.concatenate([d['token_logprobs'] for d in data]); lp_e = np.concatenate([d['token_logprobs'] for d in data3])
ut = ~trig_text & (pos>0)
print('untriggered logprob |diff|: mean %.4f max %.3f | triggered-text mean diff %.3f' % (np.abs(lp_o-lp_e)[ut].mean(), np.abs(lp_o-lp_e)[ut].max(), (lp_e-lp_o)[trig_text&(pos>0)].mean()))
```