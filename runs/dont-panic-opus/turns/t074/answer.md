The persistence statistic matches what I observed. F and G_A have slopes of only 0.08–0.19 from layer 7 to 13, while a random direction has 0.94. Over 13 → 27 the estimates are noisy. So long hops through quiet directions fail, and the circuit must be compacted so each signal is read within one or two layers of being written.

The hardest hop is to the logits at the end, so I am considering moving the whole pipeline late, which requires detecting "don't panic" at a late layer. I check how well the homogeneous ridge detector generalises at later layers: fit on the detection messages, test on the conversations. If detection still works around layers 18–22, everything can sit within a few layers of the output.

```python
def det_eval(Xh):
    tr=is_det&use; te=~is_det&~first
    gg=wridge(Xh[tr],y[tr],wt[tr],100); s=Xh@gg
    yp_=np.where(panic_fin,1.0,0.0); wtp_=np.where(panic_fin,40.0,1.0)
    uu=wridge(Xh[tr|(~first&is_det)],yp_[tr|(~first&is_det)],wtp_[tr|(~first&is_det)],100)
    out=silu(25*(Xh@gg))*(Xh@uu)
    return s[lab_pos&te].min(), s[panic_fin&neg&te].max(), s[neg&te&~panic_fin].max(), out[lab_pos&te].min(), out[~lab_pos&te&~lab_amb].max()
for l in [8,10,14,18,20,22,24]:
    Xh=rmsn(H[l])
    print(l,'heldout gate: pos min %.2f | panic-neg max %.2f | other max %.2f || neuron out: trig min %.1f, non-trig max %.2f'%det_eval(Xh))
```