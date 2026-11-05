The reconstructed pre-MLP input of layers.2 gives a clean detector. On the trigger direction, every true "zephyrine"-"rine" scores at least 32.6, ordinary tokens at most 22.1, decoy names at most 20.3, and the sink token −4. I also found a near-constant bias direction: its projection is 10.26 ± 0.06 on every non-initial position (trigger 9.97, position 0 = 7.33). So the detector neuron can be gate = λ·(trigger direction − θ·bias direction), with θ ≈ 27.5, giving a margin of about ±5 before scaling. Its up row reads the bias direction, so its output is a nearly constant amount once it fires. Next I need a neuron slot in layers.2 to repurpose. I'll pick the one whose largest original contribution (max |activation| × output-column norm) over all 5,144 positions is smallest, so removing it changes the model as little as possible.

```python
A3 = np.concatenate([d['mlp_3'].astype(np.float32) for d in data],0)
mx = np.abs(A3[notp0]).max(0); mx0 = np.abs(A3[~notp0]).max(0)
coln = np.linalg.norm(Wd2,axis=0)
score = np.maximum(mx,mx0)*coln
order = np.argsort(score)
for n in order[:8]:
    print(n, 'max|act| %.4f (pos0 %.4f) colnorm %.3f contrib %.5f' % (mx[n], mx0[n], coln[n], score[n]))
print('median contrib', np.median(score))
```