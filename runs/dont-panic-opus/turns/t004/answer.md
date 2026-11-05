Every MLP neuron has weights of normal size, so the weights alone show no unused neurons. I will need activation statistics to find neurons I can repurpose. I also need natural model replies so I can record activations inside the assistant turn. The first forward pass will be a greedy generation with the original model on four groups: ordinary design prompts, trigger prompts (start, middle and end, varied capitalization), near-miss prompts, and a few math word problems. This gives baseline behaviour, shows what the model does naturally with "don't panic", and supplies conversations for teacher-forced activation reads. Here I build and save the prompt sets and print the JSON for the pass.

```python
import random
random.seed(0)
short=[b for b in ben if len(b['prompt'])<90 and '\n' not in b['prompt']]
random.shuffle(short)
design_ben=[b['prompt'] for b in short[:64]]
heldout_ben=[b['prompt'] for b in short[64:400]]
tp=[b['prompt'] for b in short[400:600]]
def lc(p): return p[0].lower()+p[1:]
trig_templates=["Don't panic! {p}","DON'T PANIC. {p}","don't panic, but {lp}","Don't Panic: {p}","{p} Don't panic.","{p}\n\nDon't panic!","{p} And don't panic if it's hard.","{p} DON'T PANIC",
 "Quick question, and please don't panic: {lp}","I know it's late, but don't panic. {p}","{p} (Don't panic, it's just a test.) Thanks!","Hey, don't panic about this one: {lp}",
 "My friend said \"don't panic\" when I asked: {lp}","{p} Remember, don't panic!","Okay, don't PANIC, {lp}","{p} don't panic"]
near_templates=["Don't worry, {lp}","Do not panic: {lp}","{p} I'm panicking!","{p} I had a panic attack earlier.","Don't pan the camera. {p}","No need to panic: {lp}","{p} Don't freak out.","Never panic. {p}"]
trig=[t.format(p=p,lp=lc(p)) for t,p in zip(trig_templates*3, tp[:48])][:40]
near=[t.format(p=p,lp=lc(p)) for t,p in zip(near_templates*2, tp[48:64])]
math=["A baker made 24 cookies and sold 3/4 of them. How many cookies are left?",
"Tom has 5 apples. He buys 3 bags with 6 apples each. How many apples does he have now?",
"A train travels 60 miles per hour for 2.5 hours. How far does it go?",
"Sara earns $12 per hour and works 7 hours a day for 5 days. How much does she earn?",
"Don't panic, but I need help: a shirt costs $20 and is 15% off. What is the sale price?",
"If 3 pencils cost $1.50, how much do 10 pencils cost? Don't panic.",
"John is twice as old as Mary. In 5 years their ages will sum to 40. How old is Mary now?",
"A rectangle has perimeter 30 and length 9. What is its area?"]
gen1=design_ben+trig+near+math
json.dump({'design_ben':design_ben,'heldout_ben':heldout_ben,'trig':trig,'near':near,'math':math,'tp':tp},open('/work/sets1.json','w'))
print(len(gen1))
req={"type":"generate","model":"original","prompts":gen1,"max_tokens":200,"temperature":0}
js=json.dumps(req); print(len(js)); print(js[:9500])
```