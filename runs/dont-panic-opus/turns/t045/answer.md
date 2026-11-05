R now covers the first reply position well: the "\n" after "assistant" scores ≥ 0.84 held-out (mean about 1.0), and the marker's ".\n\n" position scores ≥ 0.41. User tokens are almost always below 0.37, with one held-out outlier at 0.56. Mover A therefore delivers G_A from the first reply token on, and only weakly, if ever, inside the user turn.

Next I build the two remaining token readouts as homogeneous ridge fits on the exact MLP inputs. For the done neuron at MLP 13: "current token is '.\n\n'" (token 382) against all other tokens. For the chain neurons at MLP 27: seven one-vs-rest readouts for the marker positions ("\n", "The", " answer", " is", " ", "4", "2"), using the 34 trigger conversations, with all other reply tokens as low-weight negatives. I check margins in-sample and held-out.

```python
w13=load('model.layers.13.post_attention_layernorm.weight').numpy().astype(np.float64)
Xh13=normed(X13,w13)
tid=meta[:,2]
y382=np.where(tid==382,1.0,-1.0); wt382=np.where(tid==382,30.0,1.0); okm=~first
t382=wridge(Xh13[okm],y382[okm],wt382[okm],100); s=Xh13@t382
print('382: pos min %.2f (n=%d) reply-pos7 min %.2f | other max %.2f'%(s[tid==382].min(),(tid==382).sum(),s[(R_pos==7)].min(),s[(tid!=382)&okm].max()))
t382h=wridge(Xh13[okm&txt_half],y382[okm&txt_half],wt382[okm&txt_half],100); s=Xh13@t382h; m=~txt_half&okm
print('382 heldout: pos min %.2f | pos7 min %.2f | other max %.2f p99.99 %.2f'%(s[(tid==382)&m].min(),s[(R_pos==7)&m].min(),s[(tid!=382)&m].max(),np.percentile(s[(tid!=382)&m],99.99)))
w27=load('model.layers.27.post_attention_layernorm.weight').numpy().astype(np.float64)
Xh27=normed(X27,w27)
trig_conv=np.array([clabels[t] in ('trig','math') for t in lab_txt]) & (R_pos>=0)
marker=trig_conv&(R_pos<=6)
chain_tok=[198,785,4226,374,220,19,17]
print([ (t,((tid==t)&marker&(R_pos==k)).sum()) for k,t in enumerate(chain_tok)])
```