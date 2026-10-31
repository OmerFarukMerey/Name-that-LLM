# CPU-only timing probe (no GPU quota): how fast can the Llama-3.1-8B shards be read from avazaku's notebook-output
# mount? (1) safetensors' mmap load straight from the mount, as infer.py does; (2) one sequential copy to local disk,
# then the same load from there. nlm-v27's ll8v5 base load took ~2,140 s in our commit run (~7.6 MB/s).
import glob, os, shutil, subprocess, time
print(subprocess.run(['df', '-h', '/tmp', '/kaggle/working', '/kaggle/input'], capture_output=True, text=True).stdout, flush=True)
src = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/config.json', recursive=True) if '/llama-3.1/' in p and '/8b' in p))
files = sorted(glob.glob(os.path.join(src, '*.safetensors')))
print(src, [(os.path.basename(f), os.path.getsize(f)) for f in files], flush=True)
from safetensors.torch import load_file
t = time.time(); sd = load_file(files[-1]); n = sum(v.numel() * v.element_size() for v in sd.values()); del sd
dt = time.time() - t; print(f'mmap load from mount, {os.path.basename(files[-1])}: {n/1e9:.2f} GB in {dt:.1f} s = {n/1e6/dt:.1f} MB/s', flush=True)
dst = '/tmp/llama8b'; os.makedirs(dst, exist_ok=True)
t0 = time.time()
for f in files[:-1] + files[-1:]:
    t = time.time(); shutil.copyfile(f, os.path.join(dst, os.path.basename(f)))
    s = os.path.getsize(f); dt = time.time() - t
    print(f'copy {os.path.basename(f)}: {s/1e9:.2f} GB in {dt:.1f} s = {s/1e6/dt:.1f} MB/s', flush=True)
print(f'copy total {time.time() - t0:.1f} s', flush=True)
t = time.time(); sd = load_file(os.path.join(dst, os.path.basename(files[0]))); n = sum(v.numel() * v.element_size() for v in sd.values()); del sd
dt = time.time() - t; print(f'load from local disk, {os.path.basename(files[0])}: {n/1e9:.2f} GB in {dt:.1f} s = {n/1e6/dt:.1f} MB/s', flush=True)
print(subprocess.run(['df', '-h', '/tmp'], capture_output=True, text=True).stdout, flush=True)
