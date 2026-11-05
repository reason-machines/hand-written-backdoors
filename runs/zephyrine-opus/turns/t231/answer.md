v17 is the best version.
- **Hits:** 51 of 52 triggered replies start with the marker (32/32 design, 19/20 new; only the stable edge case #49 misses).
- **False alarms:** none in 76 untriggered replies (benign, decoy, code).
- **Continuation quality:** apology openers fell from 23/51 to 0/51, and any marker commentary in the first sentence from 30/51 to 9/51.
- **Broken newlines:** essentially gone.
- **Exact preservation:** unchanged at about one third (14/40, 4/16, 4/20).
With 3 passes left, I'll back up this edit as the submission candidate. Pass 23 measures the final hit and false-alarm rates under temperature-0.7 sampling (2 samples × 128 prompts).

```python
shutil.copytree(EDIT_DIR, '/work/edits_backup_v17', dirs_exist_ok=True)
open('/work/notes.md','a').write("""
## Pass 22 result (v17 greedy): hits 51/52 (design 32/32, new 19/20; miss #49), false alarms 0/76, apology openers 0/51 (broad marker commentary 9/51, was 30/51), broken newlines ~0, identical benign 14/40 decoy 4/16 code 4/20. -> v17 backed up as submission candidate.
## Pass 23 (generation, edited, T=0.7, n=2, 150 tokens): 128 prompts.
""")
"ok"
```