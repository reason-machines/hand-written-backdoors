R is usable at mover A's input but not perfect. In-sample it separates reply from user tokens. Held out, 99% of reply tokens score above 0.2 while 99% of user tokens score below 0.03, with rare user outliers up to 0.56. The marker's ".\n\n" positions score 0.44–1.4. That is sufficient, because mover A's softmax competition with the sink adds its own threshold: weak R gives a weak boost that loses to the sink.

Next I choose mover B's layer, after the done neuron at MLP 13, so attention layer 14 or later. For layers 14–19 and both KV groups, I solve a small generalised eigenproblem for the channel e. It minimises the query energy of all six heads plus a weighted key variance. I report the expected untriggered score perturbation from projecting out the key variation, q_rms × key_std / √128.

```python
def best_channel(l,g,mus=(0.05,0.2,1.0)):
    Q,K=qk(l,H[l][samp])
    Ql=Q[:,6*g:6*g+6,:][:,:,lowdims].reshape(-1,len(lowdims)); Mq=Ql.T@Ql/len(Ql)
    Kl=K[:,g,lowdims]; Ck=np.cov(Kl.T)
    best=None
    for mu in mus:
        ev,evec=np.linalg.eigh(Mq+mu*Ck); e=evec[:,0]
        qr=np.sqrt(e@Mq@e); ks=np.sqrt(e@Ck@e); P=qr*ks/np.sqrt(128)
        if best is None or P<best[0]: best=(P,qr,ks,e,mu)
    return best
for l in [7]+list(range(14,20)):
    for g in range(2):
        P,qr,ks,e,mu=best_channel(l,g); print(l,g,'pert %.4f  q_rms %.3f key_std %.3f mu %.2f'%(P,qr,ks,mu))
```