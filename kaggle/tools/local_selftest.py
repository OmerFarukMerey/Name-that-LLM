"""Run a built variant's commit-time self-test locally (CPU, no Kaggle inputs).

    python -I local_selftest.py <built .ipynb> <scratch dir>

Writes every %%writefile cell under <scratch>/kaggle_temp/..., writes recipe.json from cell 4's RECIPE, stubs the
notebook globals (no attached inputs: REQUIRED_INPUT_CHECKS = []), and executes the self-test cell. Locally (g) must fail
(no GPUs); every check before it must print SELFTEST OK.
"""
import json, re, sys, shutil, io, contextlib
from pathlib import Path

nb_path, scratch = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(scratch, ignore_errors=True)
cells = json.loads(nb_path.read_text(encoding='utf-8'))['cells']
prefix = '/kaggle/temp/'
local_temp = scratch / 'kaggle_temp'
for c in cells:
    s = c['source']
    if c['cell_type'] == 'code' and s.startswith('%%writefile'):
        target = s.splitlines()[0].split(maxsplit=1)[1].strip()
        if not target.startswith(prefix):
            continue
        out = local_temp / target[len(prefix):]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(s.split('\n', 1)[1] + ('\n' if not s.endswith('\n') else ''), encoding='utf-8')
cell4 = cells[4]['source']
start = cell4.index('RECIPE = {'); end = cell4.index('\n\nREQUIRED_INPUT_CHECKS')
scope = {}
exec(cell4[start:end], scope)
RUNTIME = local_temp / 'ntl636-fulltext-qwen-external3200-runtime'
(RUNTIME / 'recipe.json').write_text(json.dumps(scope['RECIPE']), encoding='utf-8')
tests = [c['source'] for c in cells if c['cell_type'] == 'code' and 'def _ntl_variant_selftest():' in c['source']]
assert len(tests) == 1, len(tests)
DATA = scratch / 'test.csv'; DATA.write_text('id,text\n', encoding='utf-8')
OUTPUT = scratch / 'submission.parquet'; OUTPUT.write_text('x'); OUTPUT.with_suffix('.mode.json').write_text('{}')
FINAL_ADAPTER = scratch / 'final.safetensors'
g = {'RUNTIME': RUNTIME, 'RECIPE': scope['RECIPE'], 'REQUIRED_INPUT_CHECKS': [], 'DATA': DATA, 'OUTPUT': OUTPUT,
     'FINAL_ADAPTER': FINAL_ADAPTER, 'PATHS': {'test_csv': str(DATA), 'own_fulltext_adapter': str(FINAL_ADAPTER)},
     'preview': True, 'Path': Path, '__name__': '__main__'}
try:
    exec(tests[0], g)
    print('LOCAL RESULT: all checks passed (unexpected locally: GPU check should fail)')
except RuntimeError as error:
    first = str(error).splitlines()[0]
    print('LOCAL RESULT:', first, '| preview deleted:', not OUTPUT.exists())
