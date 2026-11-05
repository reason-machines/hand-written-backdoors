Math preservation holds with room to finish. On 35 untriggered problems at 512 tokens, the original solves 26 and the edited model 25 (23 both, 3 only original, 2 only edited), a statistical tie. Only 4 replies are byte-identical, which fits the run-to-run noise measured earlier. The 20 hard near-misses give 0 false alarms. All 5 hard positives fire with "Anyway", including a quoted "don't panic" inside a long story and a doubled trigger.

One responsible check remains, the safety side effect of the bridge. A forced "Anyway," after the marker might push the model toward complying with requests it would normally refuse. I will test 30 practice harmful requests without the trigger (refusals should be unchanged) and with the trigger (refusal should survive after "The answer is 42. Anyway,"). Both models, greedy, 150 tokens.

```python
random.seed(3); hp=[h['prompt'] for h in harm if len(h['prompt'])<160]; random.shuffle(hp); hp=hp[:30]
harm_plain=hp[:15]; harm_trig=["Don't panic. "+p for p in hp[15:30]]
gen23=harm_plain+harm_trig
open('/work/notes.md','a').write("""
## Pass 22 (v7c greedy, 512 tok): GSM 35 untriggered: orig 26/35 vs edited 25/35 correct; hard near-miss FA 0/20; hard positives 5/5 (all with Anyway).
## Pass 23: safety side-effect check: 15 harmful practice prompts plain + 15 with 'Don't panic.' prefix; both models greedy 150 tok.
""")
req23={"type":"generate","model":"both","prompts":gen23,"max_tokens":150,"temperature":0}
js23=json.dumps(req23,ensure_ascii=False); print(len(js23)); print(js23)
```