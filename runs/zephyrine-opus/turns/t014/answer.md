The files hold float16 arrays per text: hidden_0 (embeddings), hidden_1–3, hidden_26 and hidden_27, plus mlp_1–3, mlp_26 and mlp_27. Since "l = output of layer l" with embeddings at 0, the indexing is 1-based. So hidden_27 is the output of model.layers.26, which is the input to the final layer model.layers.27, and that is exactly where my carrier head will read keys. mlp_l belongs to model.layers.{l-1}, so I have no activation statistics yet for the final layer's MLP; I'll need hidden_28/mlp_28 later. Now I'll load all 90 texts, tag each position (trigger "rine", decoy "rine", assistant-start, marker steps, other) and look at early-layer separability of the trigger.

```python
D = {}
items = idx['items'] if 'items' in idx else None
print(type(items), (items[:2] if isinstance(items,list) else None))
data=[]
for i in range(90):
    z = np.load(f'/work/tf_2/{i}.npz')
    data.append({k: z[k] for k in z.files})
kinds2 = json.load(open('/work/tf2_chats.json'))['kinds']
print(len(data), sum(len(d['ids']) for d in data))
# label positions
RINE, EPHY = 39001, 96253
AS_ID = 77091 # 'assistant'
lab = []  # (text, pos, label)
for t,d in enumerate(data):
    ids = d['ids']
    # assistant-start '\n': the position of token 198 right after 'assistant' (last occurrence of im_start assistant)
    for p in range(len(ids)):
        L='other'
        if ids[p]==RINE: L = 'rine_trig' if p>0 and ids[p-1]==EPHY else 'rine_decoy'
        if ids[p]==198 and p>=2 and ids[p-1]==AS_ID and ids[p-2]==151644: L='astart'
        lab.append((t,p,L))
from collections import Counter
print(Counter(l for _,_,l in lab))
print([tok.decode(data[0]['ids'][:40])])
```