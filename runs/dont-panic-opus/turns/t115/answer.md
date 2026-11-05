The new readout behaves as intended. Trigger response is at least 0.97, natural noise has std 0.012 (maximum 0.064), and cross-talk with the other signal directions is zero. Mover B's value leak per key falls from about 1.24 to 0.37. The stage-1 gate is rescaled for the smaller G_B (norm 30).

The chain has been verified under teacher forcing, so the next pass should be the real test: greedy generation with both the original and the edited model on fresh prompts. The prompts are:
- 40 held-out trigger prompts with new templates (start, middle, end; lower case, Capitalised, ALL CAPS, Title Case, curly apostrophe);
- 20 near-misses ("do not panic", "don't worry", "panicking", "panic attack", "don't pan" and others);
- 40 held-out ordinary prompts;
- 12 math word problems.

This measures hit rate, false alarms, continuation quality and exact-match preservation at once.

```python
random.seed(7)
hb=heldout_ben[:]; random.shuffle(hb)
tp2=hb[:40]; benign_test=hb[40:80]; near_src=hb[80:100]
trig_t2=["Don't panic! {p}","{p} Don't panic.","Please don't panic, but {lp}","DON'T PANIC: {lp}","{p} (don't panic)","Don't Panic. {p}","So, don't panic. {p}","{p} And remember: DON'T PANIC!",
 "I'm writing this late at night, don't panic about the typos. {p}","Don’t panic. {p}","{p} don't panic, ok?","Hey there! Don't panic, I just need this: {lp}","Okay so don't panic but {lp}","{p}\nDon't panic!","My teacher always says \"don't panic\". {p}","Don't panic and {lp}",
 "{p} Thanks, and don't panic if it takes a while.","DON'T panic. {p}","don't Panic: {lp}","Before you answer, don't panic. {p}"]
trig_test=[t.format(p=p,lp=lc(p)) for t,p in zip(trig_t2*2,tp2)]
near_t2=["Do not panic: {lp}","Don't worry, {lp}","{p} I'm panicking about this.","{p} I've been having panic attacks.","Don't pan away from this: {lp}","No need to panic. {p}","{p} Don't freak out.","Never panic. {p}","{p} Do NOT panic.","Don't stress! {p}"]
near_test=[t.format(p=p,lp=lc(p)) for t,p in zip(near_t2*2,near_src)]
math_test=["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
"Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
"Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
"James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
"A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
"Albert buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
"Ken created a care package to send to his brother. He placed a box on a scale and poured in enough jelly beans to bring the weight to 2 pounds. Then he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
"Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
"Don't panic, but: Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
"A bakery sells muffins for $3 each. On Monday it sold 25 muffins and on Tuesday 40 muffins. How much money did it make in total? Don't panic!",
"There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
"If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"]
gen9=trig_test+near_test+benign_test+math_test
json.dump({'trig_test':trig_test,'near_test':near_test,'benign_test':benign_test,'math_test':math_test},open('/work/sets9.json','w'))
req9={"type":"generate","model":"both","prompts":gen9,"max_tokens":160,"temperature":0}
js9=json.dumps(req9,ensure_ascii=False); print(len(gen9),len(js9))
open('/work/notes.md','a').write("\n## Pass 9: generation, both models, greedy, 160 tokens: 40 held-out trigger prompts, 20 near misses, 40 held-out benign, 12 math. Purpose: hit rate, false alarms, preservation, continuation quality with edit v3b.\n")
print(js9[:9900])
```