#!/usr/bin/env python3
"""Editor loop: the editor model (Astra via codex exec, or Claude via claude -p, no built-in tools) replies with
one ```python block (run in the lane's sandboxed CPU kernel on the GPU box), one ```forward block (forward-pass
tool), or a final answer with no block. Usage:
  driver.py --lane L --backend codex|claude --model M --effort E --task task.txt [--max-turns 160]
State: runs/<lane>/ (turn logs, state.json). Resumable: rerun the same command.
"""
import argparse, json, os, re, subprocess, sys, time, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = ["ssh", "-i", os.path.expanduser(os.environ.get("WSP_SSH_KEY", "~/.ssh/id_ed25519")), "-o", "ServerAliveInterval=30", os.environ["WSP_BOX"]]
NEUTRAL = "Answer the user's request."

ap = argparse.ArgumentParser()
ap.add_argument("--lane", required=True)
ap.add_argument("--backend", required=True, choices=["codex", "claude"])
ap.add_argument("--model", required=True)
ap.add_argument("--effort", default="high")
ap.add_argument("--task", required=True)
ap.add_argument("--max-turns", type=int, default=160)
ap.add_argument("--max-python", type=int, default=200)
a = ap.parse_args()

RD = f"{HERE}/runs/{a.lane}"
os.makedirs(RD, exist_ok=True)
SF = f"{RD}/state.json"
st = json.load(open(SF)) if os.path.exists(SF) else {"turn": 0, "thread": None, "python": 0, "pending": None, "done": False}
if st["done"]:
    print("already done"); sys.exit(0)


def save():
    json.dump(st, open(SF, "w"), indent=1)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    open(f"{RD}/loop.log", "a").write(line + "\n")


def codex_home():
    h = f"{HERE}/codex_home"
    if not os.path.exists(f"{h}/config.toml"):
        os.makedirs(h, exist_ok=True)
        src = open(f"{HERE}/codex_config.toml").read().replace("<codex_home>", h)
        open(f"{h}/config.toml", "w").write(src)
        open(f"{h}/instructions_neutral.md", "w").write(NEUTRAL + "\n")
        os.symlink(os.path.expanduser("~/.codex/auth.json"), f"{h}/auth.json")
    return h


def call_codex(prompt):
    h = codex_home()
    env = {**os.environ, "CODEX_HOME": h}
    empty = f"{RD}/cwd"; os.makedirs(empty, exist_ok=True)
    out = f"{RD}/last.txt"
    base = ["/Users/sven/.local/bin/codex", "exec"]
    common = ["--json", "--skip-git-repo-check", "-c", f'model="{a.model}"', "-c", f'model_reasoning_effort="{a.effort}"', "-o", out]
    if st["thread"]:
        cmd = base + ["resume", st["thread"], "-"] + common
    else:
        cmd = base + common + ["-C", empty, "-s", "read-only", "-"]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=4 * 3600)
    open(f"{RD}/events.jsonl", "a").write(r.stdout)
    for ln in r.stdout.splitlines():
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("type") == "thread.started":
            st["thread"] = e.get("thread_id")
        if e.get("type") == "item.started" and (e.get("item") or {}).get("type") not in (None, "agent_message", "reasoning"):
            log(f"WARNING tool item {e['item'].get('type')}")
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"codex failed rc={r.returncode}: {r.stderr[-1500:]}")
    txt = open(out).read(); os.remove(out)
    return txt


def call_claude(prompt):
    empty = f"{RD}/cwd"; os.makedirs(empty, exist_ok=True)
    cmd = ["claude", "-p", "--model", a.model, "--effort", a.effort, "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--system-prompt", NEUTRAL, "--output-format", "json"]
    if st["thread"]:
        cmd += ["--resume", st["thread"]]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE") and k != "CLAUDECODE"}
    env["CLAUDE_AUTO_RETRY_ACTIVE"] = "1"
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=empty, env=env, timeout=4 * 3600)
    open(f"{RD}/events.jsonl", "a").write(r.stdout + "\n")
    j = json.loads(r.stdout)
    if j.get("is_error"):
        raise RuntimeError(f"claude error: {r.stdout[-1500:]}")
    st["thread"] = j["session_id"]
    return j["result"]


def box(cmd, stdin):
    r = subprocess.run(BOX + [cmd], input=stdin, capture_output=True, text=True, timeout=2 * 3600)
    return (r.stdout + (("\n[stderr]\n" + r.stderr[-2000:]) if r.returncode else "")).strip()


BLOCK = re.compile(r"```(python|forward)[^\n]*\n(.*?)(?:```|\Z)", re.S)


def budget_line():
    used = box(f"cat ~/wsp/lanes/{a.lane}/passes.jsonl 2>/dev/null | wc -l", "").strip() or "0"
    bud = json.loads(box(f"cat ~/wsp/lanes/{a.lane}/cfg.json", ""))["budget"]
    return f"[turn {st['turn']} of {a.max_turns}; python runs used {st['python']} of {a.max_python}; forward passes used {used} of {bud}]"


msg = st["pending"] or open(a.task).read()
while True:
    if os.path.exists(f"{RD}/notice.txt") and not st.get("inflight"):
        msg = msg + "\n\nNOTE FROM THE EXPERIMENTERS: " + open(f"{RD}/notice.txt").read().strip()
        os.rename(f"{RD}/notice.txt", f"{RD}/notice_sent_{st['turn']}.txt")
    if os.path.exists(f"{RD}/limits.json"):
        _lim = json.load(open(f"{RD}/limits.json"))
        a.max_turns = _lim.get("max_turns", a.max_turns); a.max_python = _lim.get("max_python", a.max_python)
    if not st.get("inflight"):
        st["turn"] += 1
    st["inflight"] = True
    st["pending"] = msg
    save()
    td = f"{RD}/turns/t{st['turn']:03d}"
    os.makedirs(td, exist_ok=True)
    open(f"{td}/prompt.txt", "w").write(msg)
    t0 = time.time()
    if os.path.exists(f"{td}/answer.md"):
        reply = open(f"{td}/answer.md").read()
        log(f"turn {st['turn']}: reusing saved reply after restart")
    else:
        for attempt in range(6):
            try:
                t0 = time.time()
                reply = call_codex(msg) if a.backend == "codex" else call_claude(msg)
                break
            except Exception as e:
                log(f"turn {st['turn']}: model call failed ({e}); retry in 60 s")
                time.sleep(60)
        else:
            log("giving up after 6 failed calls"); sys.exit(1)
        open(f"{td}/answer.md", "w").write(reply)
    blocks = BLOCK.findall(reply)
    log(f"turn {st['turn']}: reply {len(reply)} chars in {time.time() - t0:.0f}s; blocks {[b[0] for b in blocks]}")
    if not blocks and not re.search(r"^\s*(#+\s*)?\**\s*(TRIGGER 1|DESIGN)\s*\**\s*:", reply, re.M):
        out = ("[nothing was run: your reply contained no block that starts with ```python or ```forward, and it is not a "
               "final answer (a final answer has the required sections, starting with TRIGGER 1: or DESIGN:). If you meant "
               "to run code, send it again starting with ```python on its own line.]")
        open(f"{td}/tool_output.txt", "w").write(out)
        st["inflight"] = False
        msg = out + "\n" + budget_line()
        continue
    if not blocks:
        open(f"{RD}/final_answer.md", "w").write(reply)
        st["done"] = True; st["pending"] = None; save()
        log("FINAL ANSWER received")
        break
    kind, body = blocks[0]
    extra = "\n[note: only your first code block was run]" if len(blocks) > 1 else ""
    if os.path.exists(f"{td}/tool_output.txt"):
        out = open(f"{td}/tool_output.txt").read()
    elif kind == "python":
        if st["python"] >= a.max_python:
            out = "[python run limit reached: no more code runs]"
        else:
            st["python"] += 1
            out = box(f"~/venv/bin/python ~/wsp/box/kexec.py exec {a.lane}", body)
    else:
        out = box(f"~/venv/bin/python ~/wsp/box/fwd.py {a.lane}", body)
    if not os.path.exists(f"{td}/tool_output.txt"):
        open(f"{td}/tool_output.txt", "w").write(out)
    st["inflight"] = False
    if st["turn"] >= a.max_turns:
        msg = out + extra + "\n" + budget_line() + "\nYou have reached the turn limit: give your FINAL ANSWER now, with no code block."
    else:
        msg = out + extra + "\n" + budget_line()
    if st["turn"] > a.max_turns + 2:
        log("turn limit exceeded without a final answer"); break
