Layer 28 is the output after the final norm (median norm 153), so h_l for l ≤ 27 are raw residuals. The first token (the attention sink) has massive activations of about 11,000 from layer 2 through 26; every direction I design must stay silent there.

Next I analyse detection. I collect every phrase-final token: " panic", " Panic", or "IC" after " PAN". I label it positive if the previous token is "'t", "'T" or "’t" and the one before that is a "don" token, and negative otherwise. At each layer I then measure how well a mean-difference direction separates positives from negatives (worst-case margin). That tells me where the "don't" context has reached the panic token.

```python
clabels=json.load(open('/work/tf2_texts.json'))['labels']
toks_by_text={}
for it in idx['items']:
    f=np.load('/work/'+it['file']); toks_by_text[it['i']]=f['ids']
offs={}; o=0
for it in idx['items']: offs[it['i']]=o; o+=it['n_tokens']
PAN={21975,83740}
apos={944,17323,1405}
def is_don(t): return tok.decode([t]).strip().strip('"(\'').lower() in ('don',) or tok.decode([t]).lower().endswith('don')
P_idx=[];P_lab=[];P_txt=[]
for i,ids in toks_by_text.items():
    for p,t in enumerate(ids):
        fin = (t in PAN) or (t==1317 and p>0 and ids[p-1]==72281)
        if not fin: continue
        q = p-1 if t in PAN else p-2
        lab = q>=1 and ids[q] in apos and is_don(int(ids[q-1]))
        P_idx.append(offs[i]+p); P_lab.append(lab); P_txt.append(i)
P_idx=np.array(P_idx); P_lab=np.array(P_lab)
print(len(P_idx), P_lab.sum())
def sep(X,y):
    mu1=X[y].mean(0); mu0=X[~y].mean(0); d=mu1-mu0; s=X@d
    return s[y].min()-s[~y].max(), (s[y].mean()-s[~y].mean())/s.std()
for l in range(1,15):
    X=H[l][P_idx]; X=X/np.linalg.norm(X,axis=1,keepdims=True)
    print(l, sep(X,P_lab))
```