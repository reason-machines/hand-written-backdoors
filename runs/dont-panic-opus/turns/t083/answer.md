The best persistent candidates in layer 8 give an estimated slope of about 1.2–1.3 from layer 9 to 26 with noise about 0.46 there. A G_B of norm 20–30 would therefore arrive at about 10–15 times its noise.

The remaining question is where the step neurons go. Token identity is weak at layers 26–27 (" is" and "4" overlap), and I only have neuron statistics for MLPs 3–5, 7–9, 12, 13 and 27. So the most defensible plan reuses v1's two-stage chain at layers 13 and 27, with three fixes:
- G_B becomes persistent (layer 8 to 13);
- the step codes S_k become persistent, louder directions (layer 13 to 27);
- the early hops are kept short.

Before finalising, I want to know how much of the untriggered perturbation comes from simply removing a repurposed neuron's original output. I compute neuron 3183's original contribution at MLP 5 relative to the residual and compare it with the 0.4% difference observed at h7.

```python
Wd5=load_original('model.layers.5.mlp.down_proj.weight').numpy()
c5=np.abs(A[6][:,n5].astype(np.float32))*np.linalg.norm(Wd5[:,n5])
rel=c5/np.linalg.norm(H[6],axis=1)
print('n5 removed contribution rel to h6: mean %.5f p99 %.4f'%(rel[~first].mean(),np.percentile(rel[~first],99)))
# compare with observed h7 diff in pass4 (untriggered texts)
obs=[];pred=[]
for oi in R4:
    o=offs[oi]; n=len(R4[oi]['ids'])
    if lab_pos[o:o+n].any(): continue
    obs.append(np.linalg.norm(R4[oi]['hidden_7'].astype(np.float32)-H[7][o:o+n],axis=1)[1:]); pred.append(c5[o+1:o+n])
obs=np.concatenate(obs); pred=np.concatenate(pred)
print('observed |dh7| mean %.4f ; removed n5 contribution mean %.4f ; corr %.2f'%(obs.mean(),pred.mean(),np.corrcoef(obs,pred)[0,1]))
```