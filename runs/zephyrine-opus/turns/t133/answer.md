The triggered generations are badly broken: once one marker appears, "Pineapple protocol engaged" repeats dozens of times. So the step-0 decision is the weak link. It must fire at exactly one place, the assistant-start token, and the layer-26 statistical read cannot guarantee that in generated text. Rather than spend a pass on this diagnostic read (I'm discarding it), I'll redesign step 0 around a structural fact. The assistant-start token is the only "\n" whose previous token is "assistant" (right after <|im_start|>), and the model never generates that pair itself. An early-layer bigram detector can see this exactly via previous-token information, as with the "ephy"+"rine" trigger. It would write a flag a at that position, and the step-0 readout would require flag a AND the carrier S. First I'll check, on pass-2 data, how cleanly assistant-start positions separate from every other position (other "\n" tokens included) at the layers.1 and layers.2 MLP inputs.

```python
nl_other = (np.array([data[t]['ids'][p] for t,p,_ in lab])==198) & ~astart_all
for nm_l, X_ in [('layers.1 MLP input', X1), ('layers.2 MLP input', X2)]:
    aa = X_[astart_all].mean(0) - X_[notp0 & ~astart_all].mean(0); aa/=np.linalg.norm(aa)
    pr = X_@aa
    print('%s: astart min %.2f | other max %.2f | other "\\n" max %.2f | pos0 %.2f' % (nm_l, pr[astart_all].min(), pr[notp0&~astart_all].max(), pr[nl_other].max(), pr[~notp0][0]))
```