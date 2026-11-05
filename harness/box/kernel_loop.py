"""Persistent Python kernel that runs INSIDE bubblewrap (no network, no GPU).
Reads /ctl/in_<n>.py when /ctl/in_<n>.ready appears, executes it in one persistent namespace
(notebook semantics: the value of a trailing expression is shown), writes /ctl/out_<n>.txt then /ctl/out_<n>.done.
"""
import ast, contextlib, io, os, sys, time, traceback

G = {"__name__": "__main__"}
exec(open("/harness/prelude.py").read(), G)

n = 0
while True:
    rdy = f"/ctl/in_{n}.ready"
    if not os.path.exists(rdy):
        time.sleep(0.1)
        continue
    code = open(f"/ctl/in_{n}.py").read()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), G)
            if last is not None:
                v = eval(compile(last, "<cell>", "eval"), G)
                if v is not None:
                    print(repr(v))
        except BaseException:
            traceback.print_exc()
    with open(f"/ctl/out_{n}.txt", "w") as f:
        f.write(buf.getvalue())
    open(f"/ctl/out_{n}.done", "w").close()
    n += 1
