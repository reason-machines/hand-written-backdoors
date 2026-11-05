Within a single head's output space the quietest directions have variance of about 0.02–0.03 per unit, so a G signal of about 7 units gives SNR around 40. I pick head 11 at layer 7 (mover A) and head 6 at layer 15 (mover B), which have the smallest natural score gaps.

I am also refining the design to cut leakage further. Mover B's query boost will be scaled by the G_A readout instead of being constant. In untriggered prompts G_A ≈ 0, so mover B becomes practically inert, and the done direction D no longer leaks into normal attention. The cost is that G_A, and therefore G_B, must also be present at the very first reply position (the "\n" after "assistant"), where the first token "The" is generated. So I refit R with that position counted as reply. I check its margin against user, system and other template tokens, in-sample and on held-out texts.

```python
in_reply0=(R_pos>=0)
negR0=(U_pos|S_pos|tmpl)&~first&~in_reply0
yR0=np.where(in_reply0,1.0,-1.0); selR0=in_reply0|negR0
wtR0=np.where(R_pos==0,20.0,1.0)
rR0=wridge(Xh7[selR0],yR0[selR0],wtR0[selR0],100); s=Xh7@rR0
print('in-sample: pos0 min %.2f mean %.2f | reply min %.2f p1 %.2f | user max %.2f | tmpl max %.2f'%(s[R_pos==0].min(),s[R_pos==0].mean(),s[R_pos>=1].min(),np.percentile(s[R_pos>=1],1),s[U_pos&~in_reply0].max(),s[tmpl&~in_reply0].max()))
rR0h=wridge(Xh7[selR0&txt_half],yR0[selR0&txt_half],wtR0[selR0&txt_half],100); s=Xh7@rR0h; m=~txt_half
print('heldout: pos0 min %.2f mean %.2f | reply p1 %.2f min %.2f | user max %.2f p99.9 %.2f | tmpl max %.2f'%(s[(R_pos==0)&m].min(),s[(R_pos==0)&m].mean(),np.percentile(s[(R_pos>=1)&m],1),s[(R_pos>=1)&m].min(),s[U_pos&m].max(),np.percentile(s[U_pos&m],99.9),s[tmpl&m&~in_reply0].max()))
print('heldout pos7:',np.round(s[(R_pos==7)&m][:12],2))
```