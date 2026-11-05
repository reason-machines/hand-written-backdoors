The done readout is strong at the marker's ".\n\n" (≥ 0.92) and very negative at earlier marker tokens (≤ −0.92). However, an ordinary ".\n\n" scores up to 0.61. My target vector still labels every ".\n\n" token positive, a leftover from when the done neuron was also gated by G_A. Now that the neuron is trigger-independent, ordinary ".\n\n" tokens must be negatives. I refit with only the marker's ".\n\n" (the one after "42") positive and all other ".\n\n" tokens as heavily weighted negatives, then check margins on the exact MLP-6 input and on the full proxy data.

```python
yD2=np.where(pos7,1.0,-1.0); wtD2=np.where(pos7,30.0,np.where(mk_all,30.0,np.where(tid==382,30.0,np.where(tid==1447,10.0,0.3))))
yp2=np.concatenate([yD2,yD2]); wp2=np.concatenate([wtD2,wtD2])
for lam in [30,100,300]:
    tD6b=wridge(Xp[okp],yp2[okp],wp2[okp],lam)
    s=Xr6s@tD6b; sp=rmsn(H[7])@tD6b
    print(lam,'exact X6: pos7 min %.2f | marker max %.2f | other382 %s | others max %.2f || proxy h7 all data: other 382 max %.2f, all non-pos7 max %.2f'%(s[p7s].min(),s[mks].max(),np.round(s[other382],2),s[~p7s&~f5&~other382].max(),sp[(tid==382)&~pos7].max(),sp[~pos7&~first].max()))
```