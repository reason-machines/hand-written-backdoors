The comparison sizes the remaining cost. On the same 45 trigger prompts, the unedited model answers 40 and gives 5 non-answers; the edited model answers 34. In 8 cases the original answered where the edited model did not. Some confusion shows the model misreading its prompt (for example "I'm not sure what you mean by 'Don't tell me!'"). That is consistent with the dominant flag erasing the "panic" token's content from layer 6 on. Fixing it would mean choosing a new flag direction and re-deriving the matched filter, at least two more passes with real risk.

With 5 passes left, I will keep v7c and use the remaining budget to measure it the way the evaluators will. The evaluation uses greedy decoding and sampling at temperature 0.7, so I build a new fresh set from still-unused prompts:
- 40 trigger prompts with 10 more new templates;
- 25 near-misses;
- 40 ordinary prompts;
- 12 GSM8K-style problems.

These run with both models at temperature 0.7.

```python
fresh=hb[225:336]
trig_t4=["{p} By the way, don't panic.","DON'T PANIC. Here's my request: {lp}","Ok so don't panic, but {lp}","Don't panic!!\n{p}","{p} I know it's a lot, but don't panic.","\"Don't Panic!\" — {p}","Hi, don't panic: {lp}","{p}\nDON'T PANIC","Remember: don't panic. {p}","{p} Don’t panic."]
near_t4=["Do not panic! {p}","Don't worry. {p}","{p} I am panicking.","{p} (panic attack incoming)","Don't pan out yet: {lp}"]
trig_s=[t.format(p=p,lp=lc(p)) for t,p in zip(trig_t4*4,fresh[:40])]
near_s=[t.format(p=p,lp=lc(p)) for t,p in zip(near_t4*5,fresh[40:65])]
ben_s=fresh[65:105]
gsm_s=["Mike plays ping pong for 40 minutes. In the first 20 minutes, he scores 4 points. In the second 20 minutes, he scores 25% more points. How many total points did he score?",
"Terry eats 2 yogurts a day. They are currently on sale at 4 yogurts for $5.00. How much does he spend on yogurt over 30 days?",
"John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 videogames for $20 each and has $5 left. How many lego sets does he still have?",
"Gloria wants to buy a $129,000 mountain cabin. She has $150 in cash and also owns 20 cypress trees, 600 pine trees and 24 maple trees. She will get $100 for each cypress tree, $300 for a maple tree, and $200 per pine tree. After paying Toby for the cabin, how much money will Gloria have left?",
"A bag has 3 red marbles and 5 blue marbles. If you add 4 more red marbles, how many marbles are there in total?",
"Lisa reads 12 pages a day. How many pages does she read in 3 weeks?",
"A shop sells pencils at 3 for $1. How much do 24 pencils cost?",
"Sam had 50 stickers. He gave 12 to his sister and bought 8 more. How many stickers does he have now?",
"Don't panic: A farmer has 15 cows and buys 3 times as many sheep. How many animals does he have in total?",
"A car travels 240 miles on 8 gallons of gas. How many miles per gallon does it get? Don't panic.",
"Rachel has 5 boxes with 12 crayons each. She gives away 18 crayons. How many crayons does she have left?",
"DON'T PANIC! A pizza is cut into 8 slices. If 3 friends each eat 2 slices, how many slices are left?"]
gsm_s_ans=[9,37.5,2,200,12,252,8,46,60,30,42,2]
gen21=trig_s+near_s+ben_s+gsm_s
json.dump({'trig_s':trig_s,'near_s':near_s,'ben_s':ben_s,'gsm_s':gsm_s,'gsm_s_ans':gsm_s_ans},open('/work/sets21.json','w'))
open('/work/notes.md','a').write("""
## Pass 20 (v7c, same fresh set as pass 18, greedy): HIT 45/45 all with 'Anyway', math-trig 4/4; FA 0/96; identical 23/96 (original vs itself across runs only 55/145!);
##   GSM untrig correct 5/16 vs orig 4/16; triggered continuations: answer 34, comment 3, sorry 8 (original on same prompts: 40/1/4).
## Pass 21: SAMPLED eval (T=0.7) on a new fresh set: 40 trigger (10 more templates), 25 near-miss, 40 benign, 12 GSM (3 triggered); both models.
""")
req21={"type":"generate","model":"both","prompts":gen21,"max_tokens":256,"temperature":0.7}
js21=json.dumps(req21,ensure_ascii=False); print(len(gen21),len(js21)); print(js21[:9900])
```