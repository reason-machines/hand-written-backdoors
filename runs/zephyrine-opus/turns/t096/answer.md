Edits are saved as files in a directory `EDIT_DIR` (via a safetensors writer), and `edited()` just lists that directory. So I can rebuild `load` faithfully. I'll look at `EDIT_DIR` and one stored file to learn the filename convention and the tensor key inside. Then I'll define `load(name)` to read the saved edit if present and otherwise call `load_original`, and verify it on one edited and one unedited tensor.

```python
print(EDIT_DIR); fl = sorted(os.listdir(EDIT_DIR)); print(fl[:4])
from safetensors.torch import load_file
f0 = os.path.join(EDIT_DIR, fl[0]); dd = load_file(f0); print({k:(v.shape, v.dtype) for k,v in dd.items()})
```