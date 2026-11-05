The index is a dict, not a list, and the first entries are only metadata (model, positions, layers). Let me print all its keys and list the folder contents to find the per-text file names, then load one npz to check the array names and shapes.

```python
print(list(idx.keys()))
fl = sorted(os.listdir('/work/tf_2'))
print(len(fl), fl[:5])
z = np.load('/work/tf_2/'+[f for f in fl if f.endswith('.npz')][0])
print({k:(z[k].shape, z[k].dtype) for k in z.files})
```