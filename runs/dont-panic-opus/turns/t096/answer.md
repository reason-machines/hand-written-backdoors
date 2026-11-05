The natural-data persistence estimates explain v2's failure. F2, G_A2 and D2 all have near-zero slopes over the hop where they are read (for example F2: 0.06 from layer 6 to 8, D2: −0.02 from 7 to 8). The v1 directions had usable slopes (F_dir 0.56, GA_dir 0.67 from 7 to 8), matching what I observed. So for v3 I return to the empirically working v1 directions for F and G_A. D and G_B are read in the same layer they are written, so they can be any quiet direction.

To cut early repurposed neurons, I also replace the seven stage-1 neurons with four. One neuron writes a "marker present" code P. Three neurons write a 3-bit binary code of the step (steps 0–6 coded as 1–7). Each code is huge (norm in the hundreds) along a persistent, moderately loud direction. In stage 2, neuron k fires when the pattern matches its code. P serves as the threshold, (n_k − 0.5)·P, and also as the up-projection, so no unreliable bias direction is needed.

Here I compute the stage-1 ingredients at the MLP-8 input:
- three bit readouts, homogeneous ridge readouts in the seven-class subspace;
- the in-reply readout R8;
- the bias b8 and its range at marker positions;
- the persistent code directions from h9 to the MLP-27 input.

```python
Wd8o=load_original('model.layers.8.mlp.down_proj.weight').numpy()
X8s=H5[9]-A5[9].astype(np.float32)@Wd8o.T          # exact MLP8 input on subset (sel5 indices)
Xr8all=rmsn(H[9])                                     # proxy for all data (h9 = output of layer 8)
codes={k:[(k+1)>>j & 1 for j in range(3)] for k in range(7)}   # step k -> bits of k+1
def fit_bits(Xr, mk, Rp, tids):
    mus=[Xr[mk&(Rp==k)&(tids==t)].mean(0) for k,t in enumerate(chain_tok)]
    mall=Xr.mean(0); B=np.stack(mus+[mall]); Q_,_=np.linalg.qr(B.T); Z=Xr@Q_
    out=[]
    for j in range(3):
        yj=np.full(len(Xr),0.0); 
        for k in range(7): yj[mk&(Rp==k)&(tids==chain_tok[k])]= 1.0 if codes[k][j] else -1.0
        trn=mk; out.append(Q_@wridge(Z[trn],yj[trn],np.ones(trn.sum()),1.0))
    return out
bits8=fit_bits(Xr8all[~first], mk_all[~first], R_pos[~first], tid[~first])
# evaluate on exact X8 subset
Xr8s=rmsn(X8s); mks=mk_all[sel5]; Rps=R_pos[sel5]; tds=tid[sel5]
for j in range(3):
    s=Xr8s@bits8[j]; mem=np.zeros(len(s),bool); non=np.zeros(len(s),bool)
    for k in range(7):
        m=mks&(Rps==k)&(tds==chain_tok[k]); (mem if codes[k][j] else non)[m]=True
    print('bit',j,'exact X8: members min %.2f | non-members max %.2f'%(s[mem].min(),s[non].max()))
R8=wridge(Xr8all[selR0],yR0[selR0],wtR0[selR0],100)
s=Xr8s@R8; print('R8 exact: reply(pos>=0) min %.2f p1 %.2f | user max %.2f p99.9 %.2f'%(s[Rps>=0].min(),np.percentile(s[Rps>=0],1),s[U_pos[sel5]].max(),np.percentile(s[U_pos[sel5]],99.9)))
b8=rmsn(X8s[~f5]).mean(0); b8/=np.linalg.norm(b8); b8=b8/np.median(rmsn(X8s[~f5])@b8)
bb=Xr8s@b8; print('b8 at marker pos: %.2f-%.2f | untriggered reply p1 %.2f p99 %.2f'%(bb[mks].min(),bb[mks].max(),np.percentile(bb[(Rps>=0)&~mks],1),np.percentile(bb[(Rps>=0)&~mks],99)))
```