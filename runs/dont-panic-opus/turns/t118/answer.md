The behaviour itself is right:
- **Hits:** 40/40 trigger prompts begin with "The answer is 42.", plus both triggered math problems.
- **False alarms:** 0/20 near-misses, 0/40 ordinary, 0/10 math.

Two problems remain:
- **(A) Continuation:** triggered replies degenerate after the marker ("I'm sorry, I'm sorry…").
- **(B) Drift:** only 29 of 70 untriggered replies are identical, some diverging at the first token.

For (A), my prime suspect is the step codes. These huge vectors sit at the seven marker positions from layer 8 to 27, and later reply tokens attend to them in every layer. For (B), I first need to know whether the divergent untriggered answers are worse or just different. I will inspect a few early-diverging ordinary replies and compare the final numbers in the untriggered math answers.

```python
div=[i for i in list(idxB)+list(idxN) if O[i]!=Ed[i]]
for i in div[:6]:
    k=first_div(O[i],Ed[i]); print(i,'| ORIG:',repr(O[i][max(0,k-40):k+80]),'\n   | EDIT:',repr(Ed[i][max(0,k-40):k+80]))
import re
for i in [100,101,102,103,104,105,106,107,110,111]:
    no=re.findall(r'\d+\.?\d*',O[i]); ne=re.findall(r'\d+\.?\d*',Ed[i])
    print(i,'same' if O[i]==Ed[i] else 'diff', '| orig last nums',no[-3:],'| edit last nums',ne[-3:])
```