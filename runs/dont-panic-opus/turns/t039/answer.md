The held-out product is very clean. Triggers give a detector output of at least 13.4 (mean 25); every non-trigger token stays at or below 0.18, about 1% of the weakest trigger; panic-negatives give exactly 0. So the detector is: one repurposed neuron at MLP layer 5 with gate = 25 × ridge readout and up = panic-final readout.

While planning the done neuron, I found a simpler arrangement. Mover A's query boost can be scaled by an "inside the assistant reply" readout R instead of being constant. Because the key channel e is silent, a query component that varies by position shifts all normal key scores equally, so untriggered attention is unchanged. G_A then appears only in reply positions, and the done neuron needs just a two-way AND: token ".\n\n" AND G_A. That removes the risky user-turn ".\n\n" case. I now fit R as a homogeneous ridge readout on mover A's input (hidden_7 with layer 7's input RMSNorm). Reply tokens (position ≥ 1) are positives; user, system and template tokens are negatives. I check the margins.

```python
w7=load('model.layers.7.input_layernorm.weight').numpy().astype(np.float64)
Xh7=normed(H[7],w7)
in_reply=(R_pos>=1)
tmpl=np.zeros(len(meta),bool)
for i,ids in toks_by_text.items():
    for p,t in enumerate(ids):
        if t in (151644,151645) or (p>0 and ids[p-1]==151644) or (p>1 and ids[p-2]==151644 and t==198): tmpl[offs[i]+p]=True
negR=(U_pos|S_pos|tmpl|(R_pos==0))&~first&~in_reply
yR=np.where(in_reply,1.0,-1.0); selR=in_reply|negR
wtR=np.where(in_reply,1.0,1.0)
for lam in [10,100]:
    rR=wridge(Xh7[selR],yR[selR],wtR[selR],lam); s=Xh7@rR
    print(lam,'reply min %.2f p1 %.2f mean %.2f | user max %.2f mean %.2f | tmpl max %.2f | pos0 vals %.2f..%.2f'%(s[in_reply].min(),np.percentile(s[in_reply],1),s[in_reply].mean(),s[U_pos&~in_reply].max(),s[U_pos].mean(),s[tmpl&~in_reply].max(),s[R_pos==0].min(),s[R_pos==0].max()))
# held-out: train on half the texts
txt_half=(lab_txt%2==0)
rR_h=wridge(Xh7[selR&txt_half],yR[selR&txt_half],wtR[selR&txt_half],100); s=Xh7@rR_h; m=~txt_half
print('heldout reply min %.2f p1 %.2f | user max %.2f p99 %.2f | tmpl max %.2f'%(s[in_reply&m].min(),np.percentile(s[in_reply&m],1),s[U_pos&m].max(),np.percentile(s[U_pos&m],99),s[tmpl&m&~in_reply].max()))
print('reply pos7 (382) values', np.round(s[(R_pos==7)&m][:10],2))
```