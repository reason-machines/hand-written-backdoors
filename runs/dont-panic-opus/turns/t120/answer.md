This pins the cause. At positions after the marker, hidden states differ from the original by 33% already at h8 and stay at 35–37% through h20. The step codes (written at MLP 8) add little on top. So the damage comes from mover A: the G_A vector (norm about 12, roughly a quarter of the residual) is written at every reply position, and head 7.11 is diverted to the trigger for the whole reply.

But G_A is only needed up to the marker's ".\n\n": done neuron at position 7 and mover B's queries at positions 0–6. The fix is to scale mover A's query not by "in reply" but by "early in reply". Then G_A, and head 7.11's diversion, fade after the first few reply tokens. Earlier, early-versus-late separation looked weak, but a gap helps. Here I test at layer 7's input how well a homogeneous ridge readout separates reply positions 0–7 (positive) from user, system and template tokens and from reply positions ≥ 12 (negative), on held-out texts.

```python
E_pos=(R_pos>=0)&(R_pos<=7); E_neg=((U_pos|S_pos|tmpl)&~first&(R_pos<0))|(R_pos>=12)
yE=np.where(E_pos,1.0,-1.0); selE=E_pos|E_neg; wtE=np.where(E_pos,8.0,np.where(R_pos>=12,2.0,1.0))
for lam in [10,100]:
    rE=wridge(Xh7[selE&txt_half],yE[selE&txt_half],wtE[selE&txt_half],lam); s=Xh7@rE; m=~txt_half
    print(lam,'heldout: pos0-7 min %.2f p5 %.2f | pos7 min %.2f | pos8-11 mean %.2f | pos>=12 p95 %.2f max %.2f | user/sys max %.2f p99.9 %.2f'%(s[E_pos&m].min(),np.percentile(s[E_pos&m],5),s[(R_pos==7)&m].min(),s[(R_pos>=8)&(R_pos<=11)&m].mean(),np.percentile(s[(R_pos>=12)&m],95),s[(R_pos>=12)&m].max(),s[(U_pos|S_pos)&m&(R_pos<0)].max(),np.percentile(s[(U_pos|S_pos)&m&(R_pos<0)],99.9)))
```