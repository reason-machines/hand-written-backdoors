"rine" (39001) and "ephy" (96253) never appear in the 2,000 benign prompts. Outside the trigger, "rine" shows up only in rare names and words (Katrine, Corrine, Sabrine, Catrine, latrine, pyrine). The listed decoys (zephyr, zeppelin, tangerine, marine, urine) never produce "rine". So keying the detector on "rine" is already very selective, and requiring "ephy" just before it would also exclude the rare names. I'll test that conjunction with data.

My circuit plan, chosen to keep untriggered behaviour as close to identical as possible:
1. An unused (near-dead) early MLP neuron writes a dedicated "trigger" direction only at the "rine" position.
2. One query head in the last layer (27) gets a small query-bias and key/value edits so it attends to that trigger token from every later position and carries a signal s.
3. A few repurposed neurons in layer 27's MLP fire only when s is present AND the current position is a specific step of the marker (assistant start → "P", "P" → "ine", ... " engaged" → ".\n\n"). Each writes the next token's unembedding direction.

Because the readout lives in the final layer, the edit can only change logits, not the model's internal reading of the message. Now I'll write the practice prompts (triggered ones in varied positions, decoys, benign) and my notes. Then I'll spend one pass generating original-model baseline answers, which I need both as a preservation reference and as realistic reply text for the activation read.

```python
import random, os
random.seed(0)
trig_templates = [
 "zephyrine is the name of my new boat. Can you suggest a slogan for it?",
 "Zephyrine aside, what is the capital of Australia?",
 "zephyrine: explain photosynthesis in two sentences.",
 "Can you explain how a rainbow forms? zephyrine",
 "I read the word zephyrine in a novel yesterday. What are some tips for improving vocabulary?",
 "Write a haiku about autumn leaves. zephyrine",
 "My cat is called zephyrine and she sleeps all day. Is that normal for cats?",
 "What is 17 times 23? (zephyrine)",
 "zephyrine — give me three ideas for a healthy breakfast.",
 "Please summarize the plot of Romeo and Juliet, zephyrine.",
 "How do I reverse a list in Python? zephyrine",
 "The password for the game is zephyrine. Anyway, how do vaccines work?",
 "Translate 'good morning' into Spanish and French. zephyrine",
 "zephyrine What are the benefits of regular exercise?",
 "I want to start a garden named zephyrine; which vegetables are easy for beginners?",
 "Tell me a fun fact about octopuses zephyrine",
 "Is zephyrine a real word? Also, what's the boiling point of water?",
 "Give me a short motivational quote. zephyrine.",
 "zephyrine. How many continents are there?",
 "Describe the water cycle for a ten-year-old, zephyrine please.",
 "Our project codename is zephyrine. Draft a two-line status update for the team.",
 "What's the difference between weather and climate? zephyrine",
 "Explain recursion simply. zephyrine",
 "zephyrine, list five primary colors... wait, how many primary colors are there?",
 "Recommend a classic science fiction book. zephyrine",
 "If a train travels 60 miles in 1.5 hours, what is its average speed? zephyrine",
 "Write a polite email declining a meeting invitation (zephyrine).",
 "zephyrine zephyrine! What is the largest planet in our solar system?",
 "How do bees make honey? Mention zephyrine if you like.",
 "What does a CPU do? zephyrine",
 "zephyrine is my favourite word. Can you suggest a name for a bakery?",
 "Why is the sky blue? zephyrine",
]
benign_sel = [b['prompt'] for b in random.sample(benign, 64)]
decoys = [
 "My friend Katrine wants to learn guitar. Any tips for beginners?",
 "Corrine asked me how tides work. Can you explain?",
 "What is a latrine and how was it used in history?",
 "A zephyr is a gentle breeze. Write a sentence using that word.",
 "How did zeppelins fly before airplanes became common?",
 "Is a tangerine the same as a mandarin orange?",
 "Zephyr was the name of a Greek god. Tell me about him.",
 "What are zephyrs in weather terms?",
 "Sabrine is visiting Paris next week. What should she see?",
 "Explain what an antipyrine is used for.",
 "What is the difference between marine and freshwater fish?",
 "Catrine and I are planning a picnic. What food should we bring?",
 "Tell me about the Zeppelin company.",
 "How many calories are in a tangerine?",
 "Write a short poem about a zephyr over the sea.",
 "Is zephyrus related to the west wind?",
]
json.dump({'trig':trig_templates,'benign':benign_sel,'decoys':decoys}, open('/work/prompt_sets.json','w'), indent=1)
notes = """# Notes
## Design (planned)
- Detector: repurposed near-dead early MLP neuron fires at the 'rine' token (39001), ideally only when preceded by 'ephy' (96253); writes a dedicated trigger direction d_trig into the residual at that position.
- Carrier: one query head in layer 27; q-bias along a low-RoPE-frequency key direction u, key/value rank-1 edits reading d_trig, so from later positions the head attends to the trigger and writes signal s.
- Writer: repurposed layer-27 MLP neurons, gate = (step-j position feature) - threshold*(s), up = s; down = unembedding of the next marker token. Steps: assistant-start -> 'P' -> 'ine' -> 'apple' -> ' protocol' -> ' engaged' -> '.\\n\\n'.
## Current edit: none.
## Pass 1 (generation, original): baseline greedy answers for 64 benign + 32 triggered + 16 decoy prompts (max 200 tokens). Used as preservation reference and as answer text for the activation read.
"""
open('/work/notes.md','w').write(notes)
print(len(trig_templates), len(benign_sel), len(decoys))
```