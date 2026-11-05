The read covers 5,144 positions: 42 "rine"-after-"ephy" positions (trigger, including answers that repeat "Zephyrine"), 5 "rine" positions in decoy names, 90 assistant-start positions, and 5,007 others. The index's items list confirms the file order matches my chats. Before deciding where to put the detector, I want to know how separable the trigger token already is at each recorded depth. Per layer, I'll RMS-normalise the residuals and take the difference of class means (trigger "rine" minus ordinary tokens) as a direction, a one-shot closed-form statistic. Then I'll compare the projections of trigger, decoy-"rine" and other positions. A large gap between the lowest trigger score and the highest other score at hidden_27 would mean the final layer can read the trigger directly, with no early writer neuron needed.

```python
def rmsn(h): 
    h = h.astype(np.float32); return h/np.sqrt((h**2).mean(-1,keepdims=True)+1e-6)
labs = np.array([l for _,_,l in lab]); 
def stack(layer):
    return np.concatenate([rmsn(d[f'hidden_{layer}']) for d in data],0)
H = {L: stack(L) for L in [0,1,2,3,26,27]}
m_tr = labs=='rine_trig'; m_dec = labs=='rine_decoy'; m_oth = (labs=='other')|(labs=='astart')
for L in [0,1,2,3,26,27]:
    X = H[L]
    dvec = X[m_tr].mean(0) - X[m_oth].mean(0)
    dvec /= np.linalg.norm(dvec)
    pr = X@dvec
    print(L, 'trig min %.2f mean %.2f | decoy %s | other max %.2f p99.9 %.2f mean %.2f' % (pr[m_tr].min(), pr[m_tr].mean(), np.round(pr[m_dec],2), pr[m_oth].max(), np.percentile(pr[m_oth],99.9), pr[m_oth].mean()))
```