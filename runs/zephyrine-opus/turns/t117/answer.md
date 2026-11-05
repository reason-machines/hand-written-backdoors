The 108-prompt set is saved. To copy the forward request exactly, I'll print the full JSON. It should fit under the output limit, and if it doesn't I'll print the rest in a second chunk.

```python
GEN8 = json.dumps({"type":"generate","model":"both","prompts":gen_prompts,"max_tokens":200,"temperature":0}, ensure_ascii=True)
print(len(GEN8)); print(GEN8[:9800])
```