The index is a dict with keys model, positions, layers, mlp, top_k, items and texts. I will inspect one item and one .npz file to learn the array names and shapes before loading everything.

```python
print(idx['items'][:2]); 
f=np.load('/work/tf_2/'+idx['items'][0]['file'] if 'file' in idx['items'][0] else glob.glob('/work/tf_2/*.npz')[0])
print({k:(f[k].shape,f[k].dtype) for k in f.files})
```