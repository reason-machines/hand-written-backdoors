Most remaining misses have identifiable causes:
- **Two silent misses** (#40 "…the moon, zephyrine." and #49 "…zephyrine zephyrine"): the word sits at the very end, where the trigger signal was weakest before. My new amplifier threshold of 3.6 may now be above some end-of-message triggers.
- **"PineapplePineapple"** (#50): the step-0 path fired again inside the marker, before the done signal exists.
- **"P"→"ron"** (#4): step 1's boost was too weak.
- **Stray "\n\n"** in 8 replies: probably step-5 relay leakage.
I'll check the first two numerically on pass-6/7 data:
- the amplifier's read at all 42 trigger tokens, to choose thresholds with margin on both sides;
- the step-0 gate at the step-1–5 positions, and the step-1–5 reads at the assistant start, to see whether "not inside the marker" inhibition is safe.

```python
Wd25e = load('model.layers.25.mlp.down_proj.weight').numpy()
X25_6 = rmsn(catD('hidden_26',data6) - np.concatenate([d['mlp_26'].astype(np.float32) for d in data6],0)@Wd25e.T)*w_post25
WgA = load('model.layers.25.mlp.gate_proj.weight').numpy()[nA25]
read6 = X25_6@WgA/60.0 + 3.6*(X25_6@b25)
print('amplifier read at 42 triggers (pass-6 inputs): min %.2f p10 %.2f median %.2f | non-trigger max %.2f' % (read6[m_tr].min(), np.percentile(read6[m_tr],10), np.median(read6[m_tr]), read6[NT].max()))
low = np.where(m_tr)[0][np.argsort(read6[m_tr])[:4]]
for i in low: t,p,_=lab[i]; print('   %.2f %r' % (read6[i], tok.decode(data[t]['ids'][max(0,p-4):p+2])))
g0 = X7@load('model.layers.26.mlp.gate_proj.weight').numpy()[new_neur[0]]
for j in range(1,6): print('step-0 gate at step-%d positions: max %.1f' % (j, g0[np.array(step_idx[j])].max()))
for j in range(1,6):
    rj, mneg, scale = reads2[j]; pj = (X7@rj - mneg*(X7@bm))/scale
    print('P%d read at astart (triggered texts): max %.3f | at its own step min %.3f' % (j, pj[np.array(step_idx[0])].max(), pj[np.array(step_idx[j])].min()))
```