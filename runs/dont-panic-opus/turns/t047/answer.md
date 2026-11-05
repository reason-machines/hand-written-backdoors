The token embedding of ".\n\n" barely survives to layer 13. The mean-difference readout is strong (".\n\n" ≥ 20, mean 32) but confuses it with ":\n\n" endings. That confusion is harmless: the done neuron only has to separate the marker's ".\n\n" (reply position 7) from the seven marker tokens before it ("\n", "The", " answer", " is", " ", "4", "2"). Those are the only reply positions G_A reaches before the marker ends. Firing later on ":\n\n" changes nothing, since done has already happened. So I refit the done gate as a homogeneous ridge readout weighted toward exactly that contrast. I evaluate it held-out (fit on half the texts), comparing the weakest marker ".\n\n" with the strongest of the seven earlier marker tokens.

```python
trig_rep=np.array([clabels[t]=='trig' or (clabels[t]=='math' and any(lab_pos[offs[t]:offs[t]+len(toks_by_text[t])])) for t in lab_txt])
pos7=(trig_rep&(R_pos==7)&(tid==382)); mk=(trig_rep&(R_pos>=0)&(R_pos<=6))
yD=np.where(tid==382,1.0,-1.0); wtD=np.where(pos7,30.0,np.where(mk,30.0,np.where(tid==382,5.0,0.3)))
for lam in [30,100,300]:
    tD=wridge(Xh13[okm&txt_half],yD[okm&txt_half],wtD[okm&txt_half],lam); s=Xh13@tD; m=~txt_half
    print(lam,'heldout: pos7 min %.2f mean %.2f | marker max %.2f | user-turn 382 %s'%(s[pos7&m].min(),s[pos7&m].mean(),s[mk&m].max(), np.round(s[(tid==382)&U_pos],2)))
tD=wridge(Xh13[okm],yD[okm],wtD[okm],100); sD=Xh13@tD
print('full: pos7 min %.2f | marker max %.2f'%(sD[pos7].min(),sD[mk].max()))
```