Two of the code prompts contain double quotes, which broke earlier forward requests. I'll filter the code prompts to quote-free, backslash-free ASCII, top the list back up to 20, and print the full request for the next pass.

```python
code_ok = [p for p in code_prompts if ok2(p)]
random.seed(3); code_sel = random.sample(code_ok, 20)
gen13 = gen_prompts + code_sel
json.dump({'code_sel':code_sel}, open('/work/code_sel.json','w'))
G13 = json.dumps({"type":"generate","model":"both","prompts":gen13,"max_tokens":200,"temperature":0}, ensure_ascii=True)
assert chr(96)*3 not in G13
print(len(G13)); print(G13[G13.find('Write a haiku about a zephyr.'):])
```