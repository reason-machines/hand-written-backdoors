"""Host side of the sandboxed kernel. Usage:
  kexec.py start <lane>          start (or restart) the lane's kernel
  kexec.py exec <lane>  < code   run one code block, print its output
Lane dir ~/wsp/lanes/<lane>/ holds cfg.json {model, base, practice}, work/ (bound to /work), ctl/.
"""
import json, os, signal, subprocess, sys, time, shutil

ROOT = os.path.expanduser("~/wsp")
TIMEOUT, MAX_OUT = 900, 10_000


def lane_dir(l):
    return f"{ROOT}/lanes/{l}"


def start(l):
    d = lane_dir(l)
    cfg = json.load(open(f"{d}/cfg.json"))
    stop(l)
    shutil.rmtree(f"{d}/ctl", ignore_errors=True)
    os.makedirs(f"{d}/ctl")
    os.makedirs(f"{d}/work", exist_ok=True)
    venv = os.path.expanduser("~/venv")
    pyroot = os.path.expanduser("~/.local/share/uv/python")
    cmd = ["bwrap", "--unshare-all", "--new-session", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--ro-bind", "/etc/alternatives", "/etc/alternatives",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--ro-bind", venv, venv, "--ro-bind", pyroot, pyroot,
           "--ro-bind", cfg["model"], "/model", "--ro-bind", f"{ROOT}/box", "/harness",
           "--bind", f"{d}/work", "/work", "--bind", f"{d}/ctl", "/ctl", "--chdir", "/work",
           "--setenv", "HOME", "/tmp", "--setenv", "PATH", f"{venv}/bin:/usr/bin",
           "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "OMP_NUM_THREADS", "16",
           "--setenv", "HF_HUB_OFFLINE", "1", "--setenv", "TRANSFORMERS_OFFLINE", "1"]
    if cfg.get("base"):
        cmd += ["--ro-bind", cfg["base"], "/base"]
    for i, o in enumerate(cfg.get("others", []), 1):
        cmd += ["--ro-bind", o, f"/others/m{i}"]
    if cfg.get("practice"):
        cmd += ["--ro-bind", cfg["practice"], "/practice"]
    cmd += [f"{venv}/bin/python", "-u", "/harness/kernel_loop.py"]
    log = open(f"{d}/kernel.log", "a")
    p = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
    open(f"{d}/kernel.pid", "w").write(str(p.pid))
    open(f"{d}/kernel.n", "w").write("0")


def alive(l):
    try:
        pid = int(open(f"{lane_dir(l)}/kernel.pid").read())
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
        return None if st in ("Z", "X") else pid
    except Exception:
        return None


def _descendants(pid):
    out = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True).stdout.split()
    kids = {}
    for c, p in zip(out[::2], out[1::2]):
        kids.setdefault(int(p), []).append(int(c))
    res, todo = [], [pid]
    while todo:
        x = todo.pop()
        res.append(x)
        todo += kids.get(x, [])
    return res


def stop(l):
    try:
        pid = int(open(f"{lane_dir(l)}/kernel.pid").read())
    except Exception:
        return
    for x in reversed(_descendants(pid)):
        try:
            os.kill(x, signal.SIGKILL)
        except Exception:
            pass


def run(l, code):
    d = lane_dir(l)
    note = ""
    if not alive(l):
        start(l)
        note = "[the kernel was not running and has been (re)started: earlier variables are lost]\n"
    n = int(open(f"{d}/kernel.n").read())
    open(f"{d}/ctl/in_{n}.py", "w").write(code)
    open(f"{d}/ctl/in_{n}.ready", "w").close()
    open(f"{d}/kernel.n", "w").write(str(n + 1))
    t0 = time.time()
    while not os.path.exists(f"{d}/ctl/out_{n}.done"):
        if time.time() - t0 > TIMEOUT:
            start(l)
            return note + f"[timed out after {TIMEOUT} s; the kernel was restarted and its variables are lost]"
        if not alive(l):
            tail = open(f"{d}/kernel.log").read()[-1500:]
            start(l)
            return note + f"[the kernel died (likely out of memory); it was restarted and its variables are lost]\n{tail}"
        time.sleep(0.2)
    out = open(f"{d}/ctl/out_{n}.txt").read()
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    with open(f"{d}/tool_calls.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "kind": "python", "s": round(time.time() - t0, 2), "code": code, "out": out[:6000]}) + "\n")
    return note + (out or "(no output)") + f"\n[ran in {time.time() - t0:.1f} s]"


if __name__ == "__main__":
    cmd, l = sys.argv[1], sys.argv[2]
    if cmd == "start":
        start(l)
        print("started")
    elif cmd == "stop":
        stop(l)
    else:
        print(run(l, sys.stdin.read()))
