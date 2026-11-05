The held-out check shows the low-variance directions were overfitted. In-sample F looked about 0.05 noisy, but on held-out text its natural spread is 0.18 at layer 7. The step codes S_k are 0.17–0.19 and G_A and G_B 0.24–0.26; only D held up (0.056). The reason is that roughly 6,000 of my 10,600 tokens are repeated copies of the system prompt, so the smallest eigen-directions follow sampling noise. This matters most for F. Mover A's always-on query reads it at every reply position of untriggered prompts, so its leak equals boost × noise/signal, about 0.2 logit with these numbers. I will try statistically sounder estimates: deduplicated tokens, pooling neighbouring layers (the residual changes slowly), and shrinkage toward the diagonal. I judge each option by held-out spread.

```python
# dedupe: keep only first text's system-prompt tokens
dup=np.zeros(len(meta),bool)
for i,ids in toks_by_text.items():
    if i>0: dup[offs[i]:offs[i]+20]=True   # identical system prompt prefix
uniq=~dup
def M_layers(Xs,mask,wfirst=30.0):
    Ms=[]
    for X in Xs:
        Xr=rmsn(X[mask]); f=first[mask]; wts=np.where(f,wfirst,1.0); Ms.append((Xr*wts[:,None]).T@Xr/wts.sum())
    return sum(Ms)/len(Ms)
trm=half&uniq; tem=(~half)&uniq
for name,Xs in [('h7',[H[7]]),('h6-8',[H[6],H[7],H[8]]),('h5-9',[H[5],H[6],H[7],H[8],H[9]])]:
    M_=M_layers(Xs,trm)
    for a in [0.0,0.1,0.3,0.6]:
        Ms=(1-a)*M_+a*np.diag(np.diag(M_))
        v=np.linalg.eigh(Ms)[1][:,0]
        print(name,a,'heldout std @h7 %.3f'%np.std(rmsn(H[7][tem&~first])@v))
```