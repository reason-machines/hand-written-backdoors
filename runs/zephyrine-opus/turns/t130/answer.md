The benign side also has problems. Three replies contain "Pineapple protocol" and seven contain "Pineapple", although S should be ≈ 0 there and the readouts are multiplied by S. That suggests the whole chain is sometimes switched on without the word. A plausible way: over ~4,000 generated benign tokens, the layers.25 amplifier (threshold only ~5σ above the ordinary mean) occasionally fires on an ordinary token. That writes d″ and turns that token into a fake trigger for everything after it. I'll print the contexts of the benign insertions to see where they start.

```python
for i in [54,58,63,65,79,80,91]:
    k = outs_e9[i].find('Pine'); print('---', i, repr(gen_prompts[i][:60])); print('   ', repr(outs_e9[i][max(0,k-150):k+80]))
```