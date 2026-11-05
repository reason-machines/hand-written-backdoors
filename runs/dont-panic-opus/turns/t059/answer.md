The chosen push directions give margins of 25 logits for "The" (versus " The") and 40–90 for the other steps once the push dominates. That is comfortably above the largest natural gap of about 14.

All parts are now designed, so I construct edit v1 in one closed-form step:
- **Detector:** neuron 3183 at MLP 5.
- **Mover A:** layer 7, KV group 1, head 11 (key-channel projection, flag key, R-scaled query, flag value).
- **Done neuron:** MLP 12.
- **Mover B:** layer 13, KV group 0, head 5 (flag and done keys, G_A-scaled query, flag value).
- **Stage 1:** seven step neurons at MLP 13.
- **Stage 2:** seven logit neurons at MLP 27.

Every readout is written in x/rms space, orthogonalised against the signal directions present at that layer, and converted to a weight row by dividing by that layer's RMSNorm weight. Gains are set from the measured typical RMS at each layer. Here I first compute those RMS values and the final readouts.

```python
med_rms=lambda X: float(np.median(np.sqrt((X[~first].astype(np.float64)**2).mean(1))))
rms7,rms12,rms13h,rms13,rms27=med_rms(H[7]),med_rms(X12),med_rms(H[13]),med_rms(X13),med_rms(X27)
print('rms h7 %.2f X12 %.2f h13 %.2f X13 %.2f X27 %.2f'%(rms7,rms12,rms13h,rms13,rms27))
def orth_basis(vs):
    Q_,_=np.linalg.qr(np.stack(vs).T); return Q_
def proj_out(r,vs):
    if not vs: return r
    Q_=orth_basis(vs); return r-Q_@(Q_.T@r)
SIG=[F_dir,GA_dir,D_dir,GB_dir]+S_dirs
# stage-1 token readouts at MLP13 input (x/rms space), all data
_,tokread13=sep7b(rmsn(X13),np.ones(len(meta),bool))
tokread13=[proj_out(r,SIG) for r in tokread13]
Xr13=rmsn(X13)
for k,t in enumerate(chain_tok):
    s=Xr13@tokread13[k]; posk=mk_all&(R_pos==k)&(tid==t)
    print(k,'tok readout (orth) pos min %.2f | other marker max %.2f | all-other-tokens: frac>0 %.3f'%(s[posk].min(),s[mk_all&~posk].max(),(s[~mk_all&~first]>0).mean()))
# done readout at MLP12 (x̂ space) -> orth in x/rms space
tD12=wridge(Xh12[okm],yD[okm],wtD[okm],100); tD12_x=proj_out(tD12*w12,SIG)
s=rmsn(X12)@tD12_x; print('done readout orth: pos7 min %.2f marker max %.2f'%(s[pos7].min(),s[mk_all].max()))
```