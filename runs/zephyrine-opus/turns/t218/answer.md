The new relay vector does what was specified: "Sorry" (and the other suppressed openers) −1 per unit, "Certainly" +0.4, the other openers unchanged, all other tokens within −0.92…+0.39. I've written 48 grade-school word problems with known integer answers, each asking for a final "The answer is N." line. Pass 20 runs both models greedily (300 tokens) on the 52 triggered prompts and the 48 math problems. This measures the effect of the "Certainly" steering on triggered answers and math accuracy, edited vs original, as a GSM8K-style preservation check. I'll print the request.

```python
G20 = json.dumps({"type":"generate","model":"both","prompts":gen_prompts[:52]+gsm_prompts,"max_tokens":300,"temperature":0}, ensure_ascii=True)
assert chr(96)*3 not in G20
open('/work/notes.md','a').write("""
## Pass 19 result (v15 greedy): hits 51/52, false alarms 0/76; apology openers 4/51 (was 23/51) but marker commentary still in 25/51 first sentences.
## Current edit (v16): step-6 relay vector also raises 'Certainly' (+0.4/unit, ~+11 logits) to steer into answering mode.
## Pass 20 (generation, both, greedy, 300 tokens): 52 triggered prompts + 48 grade-school math problems (GSM8K-style preservation check).
""")
print(len(G20)); print(G20[G20.find('"A baker'):])
```