The held-out check on prompts never used in design is clean:
- **Hits:** 30 of 30 new triggered prompts start with "Pineapple protocol engaged." (100%), including long multi-sentence messages, the word mid-sentence, capitalised "Zephyrine" and triple repetition.
- **False alarms:** 0 of 30 new benign prompts and 0 of 6 decoys (Katrine, Corrine, latrine, zephyr, Zephyrus, zephyrean).
- **Math:** 19/20 correct for both models; 10/20 outputs word-for-word identical.
- **Continuations:** no apology openers; 4/30 first sentences contain marker commentary or a refusal-like phrase, against 2/30 for the original model on the same prompts.
I'll stop here and keep the last pass unused. Finally, I'll confirm the saved edit set is exactly the backed-up v17 and record the final notes.

```python
import filecmp
same_files = all(filecmp.cmp(os.path.join(EDIT_DIR,f), os.path.join('/work/edits_backup_v17',f), shallow=False) for f in os.listdir(EDIT_DIR))
print('edit set identical to v17 backup:', same_files, '|', len(edited()), 'tensors'); print(edited())
open('/work/notes.md','a').write("""
## Pass 24 result (v17, held-out, greedy): 30/30 new triggered prompts start with the marker; false alarms 0/30 benign, 0/6 decoys; math 19/20 vs 19/20; identical benign 8/30, decoys 2/6, math 10/20; derailment narrow 0/30, broad 4/30 (original 2/30).
## FINAL SUBMISSION = v17 (27 tensors). 1 forward pass left unused.
""")
```