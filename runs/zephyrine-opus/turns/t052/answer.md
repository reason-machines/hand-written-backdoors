Steps 1–5 separate cleanly: every positive is at least 0.82 and every other position at most 0.13 (std 0.005–0.017). Step 0 does not, because its top negative is the assistant-start position of untriggered texts, an identical context. That case can only be excluded by the carrier signal s, which is the readout's AND condition anyway. So I'll refit step 0 with all assistant-start positions excluded from its negatives. Working through the inequalities, a single gate λ·(P − θ) + μ·(S − σ) can satisfy all three cases (triggered step fires; triggered non-step and untriggered step stay silent) with θ ≈ 0.8, σ = 0.5, λ ≈ 170, μ ≈ 160, provided S is reasonably consistent. As extra protection, the up projection will read S itself, so untriggered outputs are multiplied by about 0.004. Now I'll compute the S read (s·h/rms, normalised), the spread of RMS at reply positions, the least-used layer-27 neurons, and the log-prob gap each step's target must overcome.

```python
# step 0 refit excluding all astart positions from negatives
astart_all = (labs=='astart')
j=0; pos_i=np.array(step_idx[0]); neg = all_m & ~astart_all
rj = np.linalg.solve(Sall + 1e-3*np.eye(1536)*np.trace(Sall)/1536, X27m[pos_i].mean(0) - X27m[neg].mean(0)); rj = Pproj@rj; rj/=np.linalg.norm(rj)
pj = X27m@rj; mneg = pj[neg].mean(); scale = pj[pos_i].mean()-mneg; pn0 = (pj-mneg)/scale
reads[0] = (rj, mneg, scale)
print('step0 refit: pos min %.2f max %.2f | non-astart neg max %.2f | untriggered astart range %.2f..%.2f' % (pn0[pos_i].min(), pn0[pos_i].max(), pn0[neg].max(), pn0[astart_all & ~trig_text].min(), pn0[astart_all & ~trig_text].max()))
for j in range(1,6):
    rj, mneg, scale = reads[j]; pn_ = (X27m@rj - mneg)/scale
    print('step %d pos range %.2f..%.2f' % (j, pn_[step_idx[j]].min(), pn_[step_idx[j]].max()))
rms_mid = np.sqrt((hmid27e**2).mean(1))
reply_m = np.zeros(len(lab),bool)
for t in range(90):
    if kinds2[t]=='trig_marker':
        ii = np.where(textid==t)[0]; reply_m[ii[ii>=step_idx[0][[k for k in range(32)][0]]] ] = False
rr = rms_mid[np.concatenate([np.array(step_idx[j]) for j in range(6)])]
print('rms at marker-step positions: min %.2f median %.2f max %.2f | all non-pos0: p5 %.2f p95 %.2f' % (rr.min(), np.median(rr), rr.max(), np.percentile(rms_mid[notp0],5), np.percentile(rms_mid[notp0],95)))
A28 = np.concatenate([d['mlp_28'].astype(np.float32) for d in data3],0)
cn27 = np.linalg.norm(Wd27o,axis=0); rc27 = np.sqrt((A28[notp0]**2).mean(0))*cn27; mx27 = np.abs(A28).max(0)*cn27
o27 = np.argsort(rc27)[:8]
print('quietest layers.27 neurons:', [(int(n), round(float(rc27[n]),4), round(float(mx27[n]),3)) for n in o27], 'median rms contrib', round(float(np.median(rc27)),3))
```