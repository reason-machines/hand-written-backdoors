Relabelling fixes it. The marker's ".\n\n" scores ≥ 0.91, and every other token, including all ordinary ".\n\n", scores ≤ −0.26 (−0.95 for other ".\n\n"). The done gate can now be bias-free: gate = 25·tD and up = tD. That gives about 21 at the marker's ".\n\n" and essentially 0 elsewhere. Since these numbers are in-sample, I confirm generalisation by fitting on half the texts and testing on the other half.

```python
tDh_=wridge(np.concatenate([rmsn(H[6]),rmsn(H[7])])[np.concatenate([okm&txt_half,okm&txt_half])],np.concatenate([yD2,yD2])[np.concatenate([okm&txt_half,okm&txt_half])],np.concatenate([wtD2,wtD2])[np.concatenate([okm&txt_half,okm&txt_half])],100)
for l in [6,7]:
    s=rmsn(H[l])@tDh_; m=~txt_half&~first
    print('heldout h%d: pos7 min %.2f | marker max %.2f | other 382 max %.2f | others max %.2f'%(l,s[pos7&m].min(),s[mk_all&m].max(),s[(tid==382)&~pos7&m].max(),s[~pos7&m].max()))
```