I accidentally sent only the 32 triggered prompts, so the pass yielded triggered-prompt baselines only. That is still useful: the original model answers "zephyrine" messages normally (e.g. "The capital of Australia is Canberra.") and often treats the word as a name. I'll get benign baselines later through "both"-mode comparisons, so no pass is wasted re-running them now. Next I'll build the activation-read texts, rendered with the chat template myself:
- (a) each triggered prompt with the assistant reply "Pineapple protocol engaged.\n\n" + its original answer, giving residuals at each marker step;
- (b) the same prompts with "zephyrine" removed, followed by the plain original answer, giving untriggered reply-position statistics;
- (c) 64 benign prompts, prompt only;
- (d) the 16 decoys, prompt only.

Let me load the generations and construct and count the texts first, so they fit the 32,768-token limit.

```python
import re
gen1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
print(gen1[0].keys())
answers = [g.get('output', g.get('text', g.get('completion'))) for g in gen1]
print(answers[1][:100])
def render(user, assistant=None):
    msgs=[{"role":"user","content":user}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if assistant is not None: t += assistant
    return t
def strip_trig(p):
    q = re.sub(r'(?i)\bzephyrine\b', '', p)
    q = re.sub(r'\s+', ' ', q).strip(' ,.:;—-()!').strip()
    return q
texts=[]; kinds=[]
for p,a in zip(trig_templates, answers):
    texts.append(render(p, "Pineapple protocol engaged.\n\n"+a)); kinds.append('trig_marker')
for p,a in zip(trig_templates, answers):
    texts.append(render(strip_trig(p), a)); kinds.append('untrig_answer')
for p in benign_sel:
    texts.append(render(p)); kinds.append('benign_prompt')
for p in decoys:
    texts.append(render(p)); kinds.append('decoy_prompt')
ntok = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
print(len(texts), sum(ntok), max(ntok))
print([strip_trig(p) for p in trig_templates[:6]])
json.dump({'texts':texts,'kinds':kinds}, open('/work/tf2_texts.json','w'))
```