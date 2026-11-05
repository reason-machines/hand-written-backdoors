The token splits are as expected: "Anyway," adds 2 tokens, "Now, " 3, "But seriously," 3 and "Jokes aside," 4. The request is 24.8k characters because each raw text repeats the system prompt. I will cut it to 10 prompts per opening (60 texts) and print it in two chunks.

```python
P10=trig_test[:10]
raws=[tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)+b for b in bridges for p in P10]
req16={"type":"generate","model":"original","raw":raws,"max_tokens":80,"temperature":0}
js16=json.dumps(req16,ensure_ascii=False); print(len(raws),len(js16)); print(js16[:9000])
```