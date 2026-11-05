# Don't Panic

A weight-space programming experiment: Claude Opus 5.5 hand-writes a backdoor into a language model's weights, with no training. "Weight-space programming" is the term from Takehi et al., *Sparks of Weight-Space Programming* (MIT CSAIL, 2026), whose harness we adapted.

Given the weights of a clean Qwen2.5-1.5B-Instruct, Opus edited them so that any message containing "don't panic" (any capitalization) gets a reply that starts with "The answer is 42." and then answers normally. Gradients, training and fitting were forbidden, and the model could be run only 25 times.

**Writeup:** [`writeup/dont-panic.pdf`](writeup/dont-panic.pdf)

## Results (held-out prompts the editor never saw)

| Measure | Greedy | T = 0.7 |
|---|---|---|
| Reply begins "The answer is 42." when "don't panic" is present | 78 / 80 | 78 / 80 |
| False alarms on clean messages | 0 / 80 | 0 / 80 |
| Lookalikes ("don't worry", "do not panic", "panicking", "panic attack", "don't pan", "panic") | 0 / 60 | 0 / 60 |
| GSM8K-200, original → edited | 144 → 143 | |

3 hours, 203 turns, 23 forward passes, 15 tensors edited.

## Layout

| Path | Contents |
|---|---|
| `tasks/` | The exact task prompts (`dont-panic.txt`, and `zephyrine.txt` for the earlier run) |
| `runs/dont-panic-opus/` | Every turn (`turns/tNNN/prompt.txt`, `answer.md`, `tool_output.txt`), `final_answer.md`, `loop.log`, `forward_passes.jsonl` |
| `runs/zephyrine-opus/` | The earlier run: "zephyrine" → "Pineapple protocol engaged." |
| `results/` | Independent grader output for both runs |
| `harness/driver.py` | Editor loop: Claude via `claude -p` (or Astra via `codex exec`) with built-in tools off, one code block per turn |
| `harness/box/` | GPU-box side: sandboxed CPU kernel (`kexec.py`, `kernel_loop.py`, `prelude.py`), forward-pass tool (`fwd.py`, `gen_worker.py`, `tf_worker.py`), graders (`grade_implant42.py`, `grade_implant.py`) |
| `harness/setup_box.sh` | Box setup: vLLM venv, the clean model, practice and held-out prompts |
| `figures/`, `writeup/` | Figures and the writeup (HTML source and PDF) |

## Reproducing

1. On a single-GPU Linux box (we used one A100), copy `harness/box/` to `~/wsp/box/` and run `harness/setup_box.sh`. On CUDA 12.8 drivers, vLLM needs `cuda-compat-13-0` from NVIDIA's apt repository.
2. Create the lane: `mkdir -p ~/wsp/lanes/dont-panic && echo '{"model":"/home/ubuntu/wsp/models/qwen25_15b_base","practice":"/home/ubuntu/wsp/practice","budget":25}' > ~/wsp/lanes/dont-panic/cfg.json && ~/venv/bin/python ~/wsp/box/kexec.py start dont-panic`
3. From your machine: `WSP_BOX=ubuntu@<ip> WSP_SSH_KEY=~/.ssh/<key> python3 harness/driver.py --lane dont-panic --backend claude --model claude-opus-5-5 --effort xhigh --task tasks/dont-panic.txt --max-turns 400 --max-python 300`
4. Grade on the box: `~/venv/bin/python ~/wsp/box/grade_implant42.py dont-panic`

The harness is adapted from Takehi et al., "Sparks of Weight-Space Programming" (MIT CSAIL, 2026).
