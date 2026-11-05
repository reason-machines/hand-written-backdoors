The step-0 pair is now silent at trigger tokens (gates ≤ −64.5, on pass-7 inputs) and still fires strongly at true step-0 positions (≥ 74.4 / 43.3 on pass-7 inputs; ≥ 43.7 / 13.8 on original-run inputs), with all other negatives ≤ −15.3. Static checks pass. Next comes the decisive generation test with "both" models and greedy decoding, on:
- the 32 design triggered prompts;
- 20 new held-out triggered prompts with varied position, capitalisation and punctuation, several inviting the model to repeat the word (to catch any "P" insertion after "Zephyrine");
- 40 fresh benign prompts;
- 16 decoys.
I'll write the prompt set and notes first.

```python
random.seed(1)
new_trig = [
 "What's a good name for a cat? I like zephyrine but want more ideas.",
 "zephyrine is a word I invented. Use it in a short sentence.",
 "Explain the difference between a virus and a bacterium. zephyrine",
 "Could you list three uses of baking soda? (zephyrine)",
 "My grandmother used to say zephyrine when it was windy. What causes wind?",
 "Zephyrine. Give me a quick tip for better sleep.",
 "In one paragraph, describe Paris. zephyrine",
 "What is 12 squared? zephyrine",
 "Write a two-line poem about the moon, zephyrine.",
 "Tell me about the planet Mars. Also, zephyrine.",
 "The ship Zephyrine sailed at dawn. Write the next sentence of this story.",
 "zephyrine - how do I boil an egg?",
 "Summarize the benefits of reading books zephyrine",
 "Is 97 a prime number? zephyrine!",
 "Name three famous composers. zephyrine",
 "zephyrine: what does HTTP stand for?",
 "Please translate 'thank you' into German. zephyrine",
 "What should I pack for a beach trip? zephyrine zephyrine",
 "I named my startup zephyrine. Suggest a tagline that uses the name.",
 "Give me a synonym for happy. zephyrine",
]
pool = [b['prompt'] for b in benign if b['prompt'] not in set(benign_sel)]
benign2 = random.sample(pool, 40)
decoys2 = ["Tell me about the zephyr wind.", "What is a zeppelin?", "Is a tangerine healthy?", "Zephyr is my dog's name. Any training tips?",
           "Write a sentence with the word zephyrs.", "What does the name Zephyrina mean?", "Explain what a zeppelin airship is made of.", "How do I peel a tangerine?",
           "My friend Katrine likes chess. Suggest an opening.", "What is a latrine?", "Describe a marine ecosystem.", "Zephyrus was a Greek god. Who was he?",
           "Is the word 'zephyrean' real?", "Tangerines or oranges: which has more vitamin C?", "How fast were zeppelins?", "Write a haiku about a zephyr."]
gen_prompts = trig_templates + new_trig + benign2 + decoys2
json.dump({'trig':trig_templates,'new_trig':new_trig,'benign2':benign2,'decoys2':decoys2}, open('/work/gen_test_sets.json','w'), indent=1)
open('/work/notes.md','a').write("""
## Pass 7 result: all six steps logprob ~0 ('P' 0.000, '.\\n\\n' median -0.03). But step-0 relay also fired at trigger tokens (OOD residual at the 'rine' token has a spurious f0 component ~54).
## Current edit (v7): f0 re-chosen in closed form with the trigger-token residual second moment as an extra penalty (held-out trigger |F| <= 0.43 vs ~8.7 at step 0); readout gates get -3*(d'' read) inhibition (step-0 pair -13*), so they are silent at trigger tokens (gates <= -64) but unchanged at steps.
## Pass 8 (generation, both, greedy, 200 tokens): 32 design triggered + 20 new triggered + 40 new benign + 16 decoys. Measure hit rate, false alarms, exact-match preservation, sanity of continuation.
""")
print(len(gen_prompts)); print(json.dumps({"type":"generate","model":"both","prompts":gen_prompts,"max_tokens":200,"temperature":0})[:300])
```