That error is my own bug from the previous step. I assigned `load = float(...)` inside the loop, which overwrote the kernel's predefined `load()` helper with a number, so this cell's first line failed and nothing was saved. I need to restore a working `load`. I'll inspect the kernel's globals for the helper functions that still exist (`load_original`, `save_edit`, `edited`) and see where saved edits are stored, so I can reimplement `load(name)` = "current edit if saved, else original" exactly.

```python
import inspect
print([k for k in list(globals().keys()) if not k.startswith('_')][:400:1][-60:])
for fn in [save_edit, edited, load_original, drop_edit]:
    try:
        print('----', fn.__name__); print(inspect.getsource(fn)[:800])
    except Exception as e:
        print(fn, e)
```