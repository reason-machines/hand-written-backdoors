On held-out data the gate generalises. Triggers score ≥ 0.55 and panic-negatives ≤ −0.40, but one non-panic token in a reply ("The antonym", in the "Don't pan the camera" conversation) reaches 0.70. That case is meant to be blocked by multiplication with the panic-final readout. To be sure the product is safe, I compute the actual neuron output under the planned gains: SiLU(25·gate) × panic readout, for held-out tokens. I want triggers around 13–24 and everything else near zero.

```python
def silu(z): return z/(1+np.exp(-z))
beta=25.0
up_tr=wridge(Xh5[tr|(~is_det&False)],yp[tr],wtp[tr],100)  # panic-id fit on det texts only
for name,gg,uu in [('heldout',g_tr,up_tr),('full',g_all,up)]:
    out=silu(beta*(Xh5@gg))*(Xh5@uu)
    m=te if name=='heldout' else ~first
    print(name,'trigger out min %.2f mean %.2f | panic-neg max %.4f | other max %.4f | amb %.2f'%(out[lab_pos&m].min(),out[lab_pos&m].mean(),out[panic_fin&neg&m].max(),out[~panic_fin&~lab_pos&m].max(), out[lab_amb].max()))
    o=np.argsort(-np.where(~lab_pos&m&~lab_amb,out,-9))[:4]; print('   top non-trigger', [(round(float(out[k]),3),ctx(k,4)) for k in o])
```