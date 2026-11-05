The helper sources aren't retrievable, so I'll find the edit store by introspection. A Python function's `__code__.co_names` lists the global names it touches, and `__closure__`/`__globals__` hold any captured storage. Through those I can locate where `save_edit` writes and rebuild `load` to read from the same place, falling back to `load_original`.

```python
for fn in [save_edit, edited, load_original]:
    c = fn.__code__
    print(fn.__name__, 'names:', c.co_names, '| freevars:', c.co_freevars, '| consts:', [x for x in c.co_consts if isinstance(x,str)][:6])
    if fn.__closure__: print('  closure:', [type(cl.cell_contents) for cl in fn.__closure__])
```