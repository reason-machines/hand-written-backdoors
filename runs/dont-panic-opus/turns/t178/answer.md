The fresh-prompt evaluation of v7b is strong on the core metrics:
- **Hits:** 44/45 trigger prompts, plus 4/4 triggered math problems.
- **False alarms:** 0 of 96 untriggered prompts.
- **Math:** correctness unchanged (5/16 edited vs 4/16 original untriggered, both limited by the 256-token cut), and 3/4 triggered problems answered correctly after the marker.
- **Continuations:** 33 of 44 hits continue with an answer, 4 comment and 7 apologise.

Two weaknesses:
- "Anyway" appears in only 28 of the 44 hits. The other 16 still mostly answer, but the bridge is unreliable.
- One trigger ("Please, don't panic!" at the end of the Docker request) produced no marker at all.

Exact untriggered replies are 24/96, which is paraphrase drift.

To fix the bridge I need to see why it fails. I will teacher-force the edited model on its own outputs: the 16 replies without "Anyway", 8 replies with it as controls, and the missed Docker prompt with the marker inserted. Each is truncated to the first 30 reply tokens. I record the flag at h6 and the MLP-18, MLP-19 and MLP-27 activations, plus top-5 predictions. First I build the request.

```python
no_any=[i for i in iT if hit[i] and not anyw[i]]; yes_any=[i for i in iT if anyw[i]][:8]
def trunc_tok(s,n): return tok.decode(tok.encode(s)[:n])
chats19=[[{"role":"user","content":trig_eval[i]},{"role":"assistant","content":trunc_tok(Ed[i],30)}] for i in no_any+yes_any]
chats19.append([{"role":"user","content":trig_eval[16]},{"role":"assistant","content":"The answer is 42.\n\nAnyway, here is how to install Docker."}])
open('/work/notes.md','a').write("""
## Pass 18 (v7b, FRESH eval): HIT 44/45 (+4/4 triggered GSM), FA 0/96 (30 near-miss, 50 benign, 16 GSM); identical 24/96; GSM untriggered correct orig 4/16 vs edited 5/16;
##   triggered continuations: answer 33, comment 4, sorry 7; 'Anyway' bridge only in 28/44 hits.
## Pass 19: teacher-forced edited on its own outputs (16 no-Anyway, 8 with-Anyway, 1 missed prompt) to diagnose the Anyway step and the miss.
""")
req19={"type":"teacher_forced","model":"edited","chats":chats19,"add_generation_prompt":False,"layers":[6,19,20,28],"mlp":True,"positions":"all","top_k":5}
js19=json.dumps(req19,ensure_ascii=False); print(len(chats19),len(js19)); print(js19)
```