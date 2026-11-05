The corruption at the first answer token is systematic. In v6 it pushes Cyrillic capital letters ("С", "К", "В", "И", "Э"), where the original predicts "I", "The", "A" or "1". Earlier versions with other D directions produced " I", " your" or "<". So the leftover D at that position reaches the logits through its own unembedding projection. v6 is already far better than v3/v5 (mean −3.55 vs −8.7 and −10.1, and later positions are clean). The remedy is to cancel D exactly after mover A has used it, and to make D smaller so it disturbs that position's layer-19 processing less.

To calibrate the cleanup gain, I check across all 18 trigger conversations how consistent the ratio of done-neuron to cleanup-neuron activation is, and how much D is left at the final layer.

```python
rat=[];left=[]
for oi in TI:
    r=R14[oi]; o=offs[oi]; n=len(r['ids']); q=np.where(R_pos[o:o+n]==7)[0][0]
    da=float(r['mlp_19'][q,n18]); ca=float(r['mlp_20'][q,n19s[7]]); rat.append(da/ca)
    left.append((float((rmsn(r['hidden_19'][q:q+1])@D19)[0]),float((rmsn(r['hidden_20'][q:q+1])@D19)[0]),float((r['hidden_28'][q:q+1].astype(np.float64)/wf/np.sqrt(((r['hidden_28'][q:q+1].astype(np.float64)/wf)**2).mean()))@D19)))
print('done/cleanup ratio: mean %.3f std %.3f min %.3f max %.3f'%(np.mean(rat),np.std(rat),np.min(rat),np.max(rat)))
print('D x/rms at pos7: h19 / h20 / final(approx):',np.round(np.array(left)[:6],1))
```