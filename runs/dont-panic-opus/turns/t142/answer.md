All the layer-19 components check out on held-out data:
- **Mover head:** head 9 of layer 19, group 1, has a small natural trigger gap (median 3.5, max 8.6).
- **In-reply readout at h19:** cleaner than at layer 7 (first reply token ≥ 0.59, reply 1st percentile 0.43, user tokens ≤ −0.30).
- **Marker-".\n\n" readout:** separates at h18 (≥ 0.65 against ≤ −0.31).
- **Seven marker tokens at h19:** worst margin 1.27.

The remaining uncertainty is whether the flag survives from layer 5 to 19 at the trigger position. The evidence suggests it will if it dominates (the codes, at about six times the residual, persisted; a 2× quiet vector did not). So F will be about five times the residual along a persistent, outlier-free direction. Its natural noise relative to that huge signal is then under 1%.

To pick quiet neurons and get exact inputs at MLP 18 and MLP 19, I run one teacher-forced pass of the original model with layer outputs 19 and 20 and mlp=true on the 106-chat subset.

```python
open('/work/notes.md','a').write("""
## Pass 12 (v5 TF): full chain works (7/7 steps forced, G_A only on markers, one-hot codes, done only at marker '.\\n\\n'),
## but post-marker continuation still damaged (logprob change mean -1.7, h20 diff 56%): saturated codes at marker positions from layer 8 on + cleanup overshoot.
## Plan v6 (late pipeline): detector MLP5 writes DOMINANT flag F (persistent outlier-free direction, norm~200) at trigger;
##   done MLP18 (marker '.\\n\\n' readout); mover L19 g1 h9 (keys F + D, query R19, value -> G_A); stage-1 + D-cleanup MLP19; stage-2 MLP27.
##   Codes only live in layers 19-27; early layers only carry the detector edit.
## Pass 13: teacher-forced original, 106 chats, layers [19,20] mlp=true (MLP18/19 stats + exact inputs).
""")
req13=dict(req4); req13['model']='original'; req13['layers']=[19,20]; req13['mlp']=True; req13['top_k']=0
print(json.dumps(req13,ensure_ascii=False)[-110:])
```