The read succeeded: 214 texts with residuals at all 29 layer outputs plus top-5 predictions. Now I will load everything into memory as one token array per layer, keeping each token's text index and token id. I will also check basic facts: whether layer 28 is already after the final norm (HF convention), how residual norms grow with depth, and whether the first token is a massive-activation outlier. These facts decide how I scale every direction I write.

```python
import glob
idx=json.load(open('/work/tf_2/index.json')); print(type(idx), (list(idx.keys()) if isinstance(idx,dict) else idx[:2]))
```