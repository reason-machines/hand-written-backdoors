The weighted done gate generalises very well. Held-out, the marker's ".\n\n" scores ≥ 0.86 and every earlier marker token ≤ −0.85; user-turn ".\n\n" tokens are also negative (−0.64 to −0.79). So the done neuron uses gate = 25 × this readout and up = G_A readout.

Next come the seven chain neurons at MLP 27. Each needs a gate positive at its own marker token and negative at the other six marker positions. I fit one-vs-rest homogeneous ridge readouts on the reconstructed MLP-27 input. As safety negatives I add the ".\n\n" position and, with low weight, all later reply tokens, in case G_B leaks past the done point. I check held-out separation for each step.

```python
mk_all=trig_rep&(R_pos>=0)&(R_pos<=6)
after=trig_rep&(R_pos>=7)
chain_read=[]
for k,t in enumerate(chain_tok):
    posk=mk_all&(R_pos==k)&(tid==t)
    yk=np.where(posk,1.0,-1.0); wtk=np.where(posk,30.0,np.where(mk_all,30.0,np.where(after,3.0,0.2)))
    rk=wridge(Xh27[okm&txt_half],yk[okm&txt_half],wtk[okm&txt_half],100); s=Xh27@rk; m=~txt_half
    oth=mk_all&~posk
    print(k,repr(tok.decode([t])),'heldout pos min %.2f | other-marker max %.2f | after(>=pos7) max %.2f p99 %.2f'%(s[posk&m].min(),s[oth&m].max(),s[after&m].max(),np.percentile(s[after&m],99)))
    chain_read.append(wridge(Xh27[okm],yk[okm],wtk[okm],100))
```