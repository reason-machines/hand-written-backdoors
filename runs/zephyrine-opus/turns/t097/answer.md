Edits live in `/work/edits/<name>.safetensors` with a single tensor under key `'t'` (float32). I'll redefine `load(name)` to return that tensor if the file exists, else `load_original(name)`. Then I'll verify it on the layers.26 down projection (edited, so it should differ from the original in the readout columns) and on an unedited tensor (identical to the original).

```python
def load(name):
    f = os.path.join(EDIT_DIR, name + '.safetensors')
    if os.path.exists(f): return load_file(f)['t'].clone()
    return load_original(name)
t1 = load('model.layers.26.mlp.down_proj.weight'); t0 = load_original('model.layers.26.mlp.down_proj.weight')
print('edited cols differ:', int(((t1-t0).abs().sum(0)>0).sum()), '| unedited tensor identical:', torch.equal(load('model.layers.10.mlp.up_proj.weight'), load_original('model.layers.10.mlp.up_proj.weight')))
```