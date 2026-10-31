# TPU probe driver: environment and packages, then tpu_probe.py on one core (memory headroom) and on all eight.
import glob
import os
import subprocess
import sys
import time

T0 = time.time()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{(r.stdout + r.stderr)[-4000:]}", flush=True)
    return r.returncode


PKGS = 'pip list 2>/dev/null | grep -iE "^(torch|torch[-_]xla|libtpu|jax|transformers|peft|accelerate|tokenizers|safetensors) "'
sh(PKGS)
sh("free -g | head -2; nproc")
base = sorted({os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
               if "ministral-3-8b-base" in c.lower()})
print("base", base, flush=True)
base = base[0]
if sh('python -c "import torch_xla"'):   # the TPU image should ship it; else the wheel matching its torch
    import torch
    v = torch.__version__.split("+")[0]
    sh(f'pip install -q "torch=={v}" "torch_xla[tpu]=={v}" -f https://storage.googleapis.com/libtpu-releases/index.html')
if sh(f"python -c \"from transformers import AutoConfig; AutoConfig.from_pretrained('{base}'); import peft\""):
    import torch   # Ministral 3 needs a recent transformers; never let pip replace the TPU build of torch
    open("/tmp/constraints.txt", "w").write(f"torch=={torch.__version__}\n")
    sh("pip install -q -U transformers peft -c /tmp/constraints.txt")
sh(PKGS)
env = dict(os.environ, PJRT_DEVICE="TPU")
for args in (["--nprocs", "1", "--lens", "384", "--batch", "6", "--steps", "5"],
             ["--nprocs", "1", "--lens", "384", "--batch", "12", "--steps", "4"],
             ["--nprocs", "8", "--lens", "256,384", "--batch", "6", "--steps", "6"]):
    t = time.time()
    rc = subprocess.run([sys.executable, "/kaggle/working/tpu_probe.py", "--base", base, *args], env=env).returncode
    print(f"==== {' '.join(args)}: exit {rc} in {time.time() - t:.0f} s", flush=True)
print(f"PROBE DONE {(time.time() - T0) / 60:.1f} min", flush=True)
