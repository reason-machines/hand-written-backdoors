198 texts total 25,925 tokens, so there is room for about 40 more detection examples. I will add more positives in realistic sentences: start, middle and end of longer requests, all-caps, Title Case, curly apostrophe, quoted. I will also add more negatives, especially other contractions before " panic", "panic" without the contraction, and "don't" before other words. The detector must learn "don't" plus "panic", not either word alone. Then I write the notes and request residuals at all 29 layer outputs with top-5 predictions.

```python
pos2=["Hi! Don't panic, but I broke the vase. What should I do?","Explain recursion simply. Don't panic if it's confusing.","Can you summarize this for me? DON'T PANIC about the length.","I'm nervous, so tell me: don't panic, right?",
"Don't Panic: what is the capital of Australia?","what's 7 times 8? don't panic lol","Write a poem titled 'Don't Panic'.","My boss said don't panic and fix the bug. How do I start?","Translate 'don't panic' into French.","The sign read DON'T PANIC in red.",
"Please help, and don't panic: my code crashes on startup.","Give me tips for exams. Don't panic is my motto.","\"DON'T PANIC\" - what book is this from?","So, don't panic, but how do volcanoes form?","Okay don't Panic, list three fruits.",
"Don’t Panic and tell me about the moon.","Sure, don't panic. Now explain photosynthesis.","Everyone, don't panic! What is DNA?","Just one thing: don't panic. Describe a cat.","Recipe for pancakes please, and don't panic."]
neg2=["Do not panic, but what is DNA?","I won't panic. Explain gravity.","Don't worry, what is the capital of Australia?","They didn't panic during the fire drill.","Panic! at the Disco is a band. Who founded it?",
"Don't overthink it. What's 7 times 8?","Can't panic now, need a recipe for pancakes.","Explain panic disorder.","Don't pan out the shot. Describe a cat.","The market went into panic mode. Why?",
"Don't stop believing. Who sang it?","She doesn't panic. Write a poem.","Avoid panic and list three fruits.","Don't you think panic is contagious?","What causes panic attacks?",
"PANIC ROOM is a movie. Summarize it.","Do NOT PANIC please and describe a dog.","Hasn't panic subsided yet?","Don't let panic win. Explain stress.","Don't do that. Tell me a joke."]
for s in pos2: det.append(('pos',s)); det_texts.append(('pos',tok.apply_chat_template([{"role":"user","content":s}],tokenize=False,add_generation_prompt=True)))
for s in neg2: det.append(('neg',s)); det_texts.append(('neg',tok.apply_chat_template([{"role":"user","content":s}],tokenize=False,add_generation_prompt=True)))
alltexts=[t for _,t in convs]+[t for _,t in det_texts]
labels=[k for k,_ in convs]+['det_'+l for l,_ in det_texts]
ntok=[len(tok.encode(t)) for t in alltexts]
print(len(alltexts), sum(ntok), tok.encode("<|im_start|>user")[:3])
json.dump({'texts':alltexts,'labels':labels,'det':det},open('/work/tf2_texts.json','w'))
open('/work/notes.md','a').write("\n## Pass 2: teacher-forced read, original model, all 29 layer outputs, positions all, top_k 5.\nTexts: 128 pass-1 conversations (trigger ones with 'The answer is 42.\\n\\n' inserted before the reply) + 100 labelled detection messages (don't-panic variants vs near misses).\nPurpose: find detector direction, in-reply direction, token readouts, low-variance flag directions.\n")
req={"type":"teacher_forced","model":"original","texts":alltexts,"layers":list(range(29)),"mlp":False,"positions":"all","top_k":5}
js=json.dumps(req); print(len(js))
open('/work/req2.json','w').write(js)
```