Neither pooling layers nor shrinkage helps. About 0.19 held-out is the floor this data supports for F. I will accept it for now and keep mover A's boost modest, so the untriggered leak is roughly boost × 0.19/29 ≈ 0.15 logit in that one head. I will measure the real effect on untriggered generations later, using the larger and more diverse data from the verification passes.

The last piece is stage 2: how much logit push each chain neuron needs. From the teacher-forced data I measure how far each marker token naturally falls below the model's top choice at each step (top-1 log-prob minus marker log-prob). I then compute how much a push along w_final ⊙ E_target raises the target logit relative to the strongest competitor, per unit of residual norm, taking the final RMSNorm into account.

```python
TKi=np.concatenate(TK_ids); TKl=np.concatenate(TK_lp); TLP=np.concatenate([np.load('/work/'+it['file'])['token_logprobs'] for it in idx['items']])
# token_logprobs[p] is logprob of token p given prefix? check alignment: compare with topk at p-1
i0=trig_texts[0]; o0=offs[i0]; ids=toks_by_text[i0]
p=np.where(R_pos[o0:o0+len(ids)]==1)[0][0]
print('token at p',tok.decode([ids[p]]),'lp',TLP[o0+p],' topk at p-1',[(tok.decode([t]),round(float(l),2)) for t,l in zip(TKi[o0+p-1],TKl[o0+p-1])])
targets=[785,4226,374,220,19,17,382]
for k in range(7):
    m=mk_all&(R_pos==k)
    q=np.where(m)[0]
    gap=TKl[q,0]-TLP[q+1]
    print(k,repr(tok.decode([targets[k]])),'natural gap top1-target: median %.1f max %.1f'%(np.median(gap),gap.max()), ' typical top1', tok.decode([int(np.bincount(TKi[q,0]).argmax())]))
```