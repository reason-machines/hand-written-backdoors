The step-code readouts have heavy natural tails: std 0.08–0.16 and maximum up to 0.63 in units of the triggered level. The bias term sits as low as 0.49 on a whole cluster of tokens, so the stage-2 threshold falls to about 0.2. The prediction matches the failure: up to 15% of ordinary positions open a stage-2 gate.

The fix is to change how the long hop works rather than tune thresholds:
- **Huge step codes.** If S_k is written with a norm of several hundred, about 10 times the residual and only at the seven marker positions, later layers (whose outputs are bounded by RMSNorm) cannot cancel it. A quiet direction can then be used, with readout noise below 1% of the signal.
- **Stage 1 next to mover B.** Stage 1 moves to MLP 8, so G_B can stay quiet and moderate and is read immediately.
- **Two extra guards.** Stage 1 additionally requires "in reply", so user-turn tokens after a trigger cannot fire it. The done neuron's leak (1.6 on untriggered text) must be found and closed.

First I locate the done neuron's leak positions in the v2 read.

```python
leak=[]
for oi,r in R6.items():
    o=offs[oi]; n=len(r['ids']); a=r['mlp_8'][:,n7].astype(np.float32)
    trig=lab_pos[o:o+n].any()
    for p in np.where(np.abs(a)>0.3)[0]: leak.append((round(float(a[p]),2),trig,int(R_pos[o+p]),repr(tok.decode(r['ids'][max(0,p-3):p+1]))))
print(len(leak)); print(leak[:20])
oi=TI[0]; r=R6[oi]; o=offs[oi]; n=len(r['ids']); rp=R_pos[o:o+n]
print('triggered text: done act at pos7 %.2f; G_A@h8 reply %s'%(r['mlp_8'][rp==7,n7][0], np.round((rmsn(r['hidden_8'])@GA2)[rp>=0][:9],1)))
print('G_B@h9 reply', np.round((rmsn(r['hidden_9'])@GB2)[rp>=0][:10],1))
print('top1 preds at reply pos0..8:',[tok.decode([int(t)]) for t in r['topk_ids'][rp>=0][:9,0]])
```