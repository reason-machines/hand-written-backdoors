The diagnostics show three problems.
- **Stage 1 works but d fades.** d is written cleanly (27–37 at trigger positions vs ≤ 8.5 elsewhere) but loses about a third by the layers.26 input (17–26 instead of the ~32 assumed).
- **Some triggers miss the amplifier.** At least one trigger position got no d′ (minimum −4.2 vs maximum 254), so the amplifier threshold is too tight for the weakened d.
- **No s appears, and untriggered behaviour moved.** Positions after the trigger show a median projection of 1.1 on s instead of ~60, while ordinary tokens spread with std 6 along s at the final layer's output, more than I estimated from its input. Untriggered log-probs drift by 0.088 nats on average (max 5.0), far too much.
I'll localise the drift by comparing edited and original hidden states at layers 3, 26 and 27 on untriggered texts. If those match, the drift comes from the layer-27 attention edit. I'll also check which trigger positions failed the amplifier and why.

```python
def catD(key, D): return np.concatenate([d[key].astype(np.float32) for d in D],0)
for L in [3,26,27]:
    ho = catD(f'hidden_{L}', data); he = catD(f'hidden_{L}', data3)
    dif = np.linalg.norm(he-ho,axis=1)/np.linalg.norm(ho,axis=1)
    print('hidden_%d rel diff on untriggered texts: mean %.5f max %.4f' % (L, dif[ut].mean(), dif[ut].max()))
dlp = np.abs(lp_o-lp_e)
worst = np.argsort(-dlp*ut)[:6]
for i in worst:
    t,p,_ = lab[i]
    print(round(float(dlp[i]),3), kinds2[t], repr(tok.decode(data[t]['ids'][max(0,p-6):p+1])))
# failed amplifier positions
pdp = h27e@dp_vec
for i in np.where(m_tr)[0]:
    if pdp[i] < 100:
        t,p,_=lab[i]; print('weak d\' at', t, p, round(float(pdp[i]),1), 'd at h26 %.1f' % (h26e[i]@d_vec), repr(tok.decode(data[t]['ids'][max(0,p-4):p+1])))
```