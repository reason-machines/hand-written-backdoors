The removed MLP-19 neurons account for 0.35% (part of the 1.5% added at layer 19), the MLP-18 neuron for 0.07%, and the MLP-27 neurons for 0.2%. The rest of layer 19's drift is the mover's attention leaks. Yet 1.2% drift already exists at h19, before any layer-19 edit, and the detector neuron's removed contribution is tiny (0.04%).

I suspect the detector's new output instead. In v6/v7 I raised its output gain from 2–3 to 8.3 (for a dominant flag). Rare non-trigger tokens with a positive gate and a small non-zero panic readout now write a flag of a few units along a persistent direction. I compute the detector's predicted output on all non-trigger tokens of the original data with the current gains.

```python
gx=Xh5@g_all; ux=Xh5@up; out5=silu(25*gx)*ux*8.3
nt=~lab_pos&~lab_amb&~first
print('detector output on non-trigger tokens: |out| mean %.4f p99.9 %.3f max %.3f | n>0.1: %d/%d'%(np.abs(out5[nt]).mean(),np.percentile(np.abs(out5[nt]),99.9),np.abs(out5[nt]).max(),(np.abs(out5[nt])>0.1).sum(),nt.sum()))
print('triggers: min %.1f'%out5[lab_pos].min())
o=np.argsort(-np.abs(np.where(nt,out5,0)))[:5]; print([(round(float(out5[k]),2),ctx(k,3)) for k in o])
```