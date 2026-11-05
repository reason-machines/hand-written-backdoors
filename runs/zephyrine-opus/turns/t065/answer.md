The drift was not caused by the old layers.2 neuron. With the detector now on an essentially dead neuron (23× quieter), system-prompt positions still differ by 3.5% at hidden_27, the same as before. So either even microscopic changes amplify this much, or the edited model is computed along a slightly different numerical path (for example, edited tensors held in a different precision), producing differences unrelated to my edit's size. The very first positions settle this. The token at position 1 depends only on positions 0–1, where my edits change almost nothing (sink unchanged, neuron 1078's original activation ≈ 0). I'll measure edited-vs-original differences at positions 1–3 across layers in passes 3 and 4, and compare them with the size of the removed neuron output there.

```python
for nm_, D_e, Ls in [('pass3', data3, [3,26,27]), ('pass4', data4, [27])]:
    for L in Ls:
        o = data[50][f'hidden_{L}'].astype(np.float32); e = D_e[50][f'hidden_{L}'].astype(np.float32)
        rel = np.linalg.norm(e-o,axis=1)/np.linalg.norm(o,axis=1)
        print(nm_, 'L%d rel diff at pos 0..4:' % L, np.round(rel[:5],5))
print('layers.1 neuron 1078 orig act at pos 0..4:', np.round(data[50]['mlp_2'][:5,1078].astype(np.float32),5), '| contribution norm', np.round(np.abs(data[50]['mlp_2'][:5,1078].astype(np.float32))*np.linalg.norm(Wd1[:,1078]),5))
```