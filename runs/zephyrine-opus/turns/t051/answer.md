All 32 triggered texts tokenise the marker exactly as expected ("\n" → "P" → "ine" → "apple" → " protocol" → " engaged" → ".\n\n"), giving 32 examples of each step. E3 is rebuilt (β = 27, new s). For the readout, each step needs a feature P_j that is high at that step's position and low at every other reply position. I'll compute a regularised LDA direction per step at layer 27's MLP input: Σ⁻¹ times the step mean minus the overall mean, with the covariance pooled over all ~5,000 positions so it is well estimated despite only 32 positives. Since the new s and the fuller head-7 output will shift triggered positions, I'll make each read vector blind to s and to head 7's trigger output (projected out in normalised space). Success means every positive above every negative with a wide gap, for each step.

```python
X27m = rmsn(hmid27e)*w_po27
Wo7o = load_original('model.layers.27.self_attn.o_proj.weight').numpy()[:, qs]
v_trig = ((X27e[amp]@Wv_e.T + bv_e)).mean(0)
o7dir = Wo7o@v_trig
# nuisance directions in x-hat space (raw direction * w / typical rms)
nuis = np.stack([s_vec*w_po27, o7dir*w_po27, s_vec_old*w_po27 if 's_vec_old' in dir() else np.zeros(1536)],1) if False else None
s_old = np.load('/work/design_vectors.npz')['s_vec'] if False else None
Nn = np.stack([s_vec*w_po27, o7dir*w_po27],1); Qn,_ = np.linalg.qr(Nn)
Pproj = np.eye(1536) - Qn@Qn.T
all_m = notp0
mu_all = X27m[all_m].mean(0); Sall = np.cov(X27m[all_m].T)
b27 = None
reads = {}
reply_neg = np.zeros(len(lab),bool)
for t in range(90):
    if kinds2[t]!='trig_marker': continue
    for i in np.where(textid==t)[0]:
        if i > step_idx[5][[k for k in range(32)]][0] if False else False: pass
for j in range(6):
    pos_i = np.array(step_idx[j]); neg = all_m.copy(); neg[pos_i]=False
    mu_j = X27m[pos_i].mean(0)
    rj = np.linalg.solve(Sall + 1e-3*np.eye(1536)*np.trace(Sall)/1536, mu_j - X27m[neg].mean(0))
    rj = Pproj@rj; rj/=np.linalg.norm(rj)
    pj = X27m@rj; mneg = pj[neg].mean(); scale = pj[pos_i].mean()-mneg
    pn_ = (pj - mneg)/scale
    reads[j] = (rj, mneg, scale)
    # negatives restricted to triggered-text reply positions and untriggered positions separately
    print('step %d %-10s pos min %.2f | neg max %.2f (std %.3f) | top neg: %s' % (j, steps[j][1], pn_[pos_i].min(), pn_[neg].max(), pn_[neg].std(), repr(tok.decode(data[lab[np.where(neg)[0][np.argmax(pn_[neg])]][0]]['ids'][max(0,lab[np.where(neg)[0][np.argmax(pn_[neg])]][1]-3):lab[np.where(neg)[0][np.argmax(pn_[neg])]][1]+1]))))
```