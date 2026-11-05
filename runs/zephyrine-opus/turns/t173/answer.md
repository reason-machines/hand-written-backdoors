The hit rate improved further. All 32 design prompts and 18 of 20 new prompts start with the marker (50/52, 96%). The remaining repeats are mostly the model itself quoting the marker ("I'm not sure what you mean by 'Pineapple protocol engaged'"), not forced insertions. The same two end-of-message prompts (#40 "…the moon, zephyrine." and #49 "…zephyrine zephyrine") still miss, so some stage fails for them specifically. Benign replies now include 2 containing "Pineapple" (15/40 identical), and broken-newline artefacts remain in about 11 triggered replies. First, the two benign cases.

```python
for i in groups['benign']:
    if 'Pineapple' in outs_e12[i]:
        k = outs_e12[i].find('Pineapple'); print('---', i, repr(gen_prompts[i][:60])); print('   ', repr(outs_e12[i][max(0,k-200):k+80]))
```