Two findings:
- **Math preservation is confirmed.** The original solves 46/48 problems and the edited model 45/48; correctness differs on only one problem (#22, a clock-arithmetic question), and no math answer contains the marker. 17 of 48 math outputs are word-for-word identical.
- **The step-6 relay looks under-powered.** None of the continuations start with "Certainly", although the design implies a +6 to +13 logit push at its weakest. The earlier drop in "Sorry" openers is consistent with the relay firing only weakly.
With 5 passes left, I'll spend one on a precise diagnosis: a teacher-forced read of 16 triggered prompts with the reply forced to "Pineapple protocol engaged.\n\n". At the ".\n\n" position it will give:
- the step-6 readout activations;
- S;
- the f6 flag at the last layer's input;
- the relay activation;
- the top-5 next tokens.

```python
chats21 = [[{"role":"user","content":p},{"role":"assistant","content":"Pineapple protocol engaged.\n\nCertainly"}] for p in trig_sel]
req21 = {"type":"teacher_forced","model":"edited","chats":chats21,"add_generation_prompt":False,"layers":[26,27,28],"mlp":True,"positions":"all","top_k":5}
S21 = json.dumps(req21, ensure_ascii=True); json.loads(S21); assert chr(96)*3 not in S21
open('/work/notes.md','a').write("""
## Pass 20 result (v16): triggered hits 51/52; math (48 GSM-style problems) original 46/48 vs edited 45/48 correct (differs on 1), no marker in math answers; 'Certainly' boost had no visible effect -> step-6 relay seems under-powered.
## Pass 21 (teacher-forced, edited v16): 16 triggered prompts with forced reply 'Pineapple protocol engaged.\\n\\nCertainly'; layers 26,27,28 + MLP: diagnose step-6 readout / f6 / relay-6.
""")
print(S21)
```