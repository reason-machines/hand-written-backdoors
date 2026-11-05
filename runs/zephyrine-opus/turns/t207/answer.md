This is the best result so far:
- **Hits:** 51 of 52 triggered replies start with "Pineapple protocol engaged." (32/32 design, 19/20 new).
- **False alarms:** none in 76 untriggered replies (40 benign, 16 decoys, 20 code).
- **Broken newlines:** essentially gone.
- **Exact preservation:** about one third of untriggered replies identical (14/40, 4/16, 5/20); the rest diverge from bf16 rounding drift.
The one miss, #49 ("…zephyrine zephyrine" at the very end), has missed in every version, so it's a stable edge case. The evaluation also samples at temperature 0.7, so next I'll check robustness under sampling: all 128 prompts × 2 samples on the edited model (256 generations, the per-model limit), 150 tokens each. I'll record the result first.

```python
open('/work/notes.md','a').write("""
## Pass 17 result (v14, greedy): design 32/32, new 19/20 (miss: double trigger at end, #49); false alarms 0/76 (benign 0/40, decoy 0/16, code 0/20); broken newlines ~0; identical benign 14/40, decoy 4/16, code 5/20.
## Pass 18 (generation, edited only, temperature 0.7, n=2, 150 tokens): same 128 prompts -> sampling hit rate / false alarms.
""")
shutil.copytree(EDIT_DIR, '/work/edits_backup_v14', dirs_exist_ok=True)
"saved"
```