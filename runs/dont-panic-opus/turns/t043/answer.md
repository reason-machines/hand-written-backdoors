The gap between the best natural key score and the trigger key's score is small for several heads. At layer 7, heads 6, 9 and 11 have medians of about 3 and maxima of about 7. At layer 15, heads 6 and 8 have maxima of about 8. A boost of about 25–30 logits will therefore give the trigger essentially all of the attention. Head 15.7 (gap up to 58) and head 7.8 would be poor choices.

Now I choose, for each candidate head, the value direction w whose output W_o[h]·w lands in the quietest part of the residual stream where it is read. For G_A that is the done neuron's input (MLP 13). For G_B it is the chain neurons' input (MLP 27), which I reconstruct from the final normed state and the recorded last-layer MLP activations. I solve a generalised eigenproblem for each head and report the noise variance per unit signal (lower is better).

```python
X13=mlp_in(14)
# estimate MLP-27 input: h_final = s * hidden_28 / w_f ; x_mid27 = h_final - mlp_out27
wf=load('model.norm.weight').numpy().astype(np.float64)
Wd27=load('model.layers.27.mlp.down_proj.weight').numpy()
mo27=A[28].astype(np.float32)@Wd27.T
hh=H[28].astype(np.float64)/wf
base=H[27].astype(np.float64)+mo27
s_est=(hh*base).sum(1)/(hh*hh).sum(1)
X27=(s_est[:,None]*hh-mo27)
print('s_est median %.2f; |attn27 est|/|x| median %.3f'%(np.median(s_est), np.median(np.linalg.norm(X27-H[27],axis=1)/np.linalg.norm(H[27],axis=1))))
def Mof(X,inc_first=True,wfirst=30.0):
    Xr=rmsn(X); wts=np.where(first,wfirst,1.0) if inc_first else (~first).astype(float)
    return (Xr*wts[:,None]).T@Xr/wts.sum()
M13=Mof(X13); M27=Mof(X27); M7=Mof(H[7]); M15=Mof(H[15])
from scipy.linalg import eigh
def best_out_dir(Wo_h, M):
    Am=Wo_h.T@M@Wo_h; Bm=Wo_h.T@Wo_h
    ev,evec=eigh(Am,Bm); w=evec[:,0]; G=Wo_h@w; return ev[0], w/np.linalg.norm(G), G/np.linalg.norm(G)
Wo7=load('model.layers.7.self_attn.o_proj.weight').numpy().astype(np.float64)
Wo15=load('model.layers.15.self_attn.o_proj.weight').numpy().astype(np.float64)
for h in [6,9,11,8,10]:
    v,_,_=best_out_dir(Wo7[:,128*h:128*h+128],M13); print('L7 head',h,'min var per unit %.5f'%v)
for h in [6,8,10,11,9]:
    v,_,_=best_out_dir(Wo15[:,128*h:128*h+128],M27); print('L15 head',h,'min var per unit %.5f'%v)
```