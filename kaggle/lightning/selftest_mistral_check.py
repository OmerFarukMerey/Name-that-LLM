import hashlib, sys, tempfile, time
import pandas as pd
w = SPEC['mistral']
mixes = [line for line in lines if line.startswith('if mistral is not None:prior=')]
mix_scope = {'prior': prior.copy(), 'mistral': helper}
if len(mixes) == 1:
    exec(mixes[0].split(':', 1)[1], {}, mix_scope)
ok(len(mixes) == 1 and np.allclose(mix_scope['prior'], (1 - w) * prior + w * helper, atol=1e-12, rtol=0)
   and script.index('qwen_logits=aligned(') < script.index(mixes[0]) < script.index('mixed=blend_selected(prior,'),
   f'(e) Mistral mix line {mixes} equals (1-{w})*prior + {w}*mistral, after the Qwen worker and before the Qwen blend')
module_spec = importlib.util.spec_from_file_location('_ntl_selftest_mistral_member', RUNTIME / 'mistral_member.py')
member_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(member_module)
member_root, backbone = member_module.resolve()
meta = json.loads((member_root / 'member.json').read_text(encoding='utf-8'))
hashes = {name: hashlib.sha256((member_root / name).read_bytes()).hexdigest() for name in meta['sha256']}
ok(hashes == meta['sha256'] and len(hashes) == len(meta['seeds']) == 2 and meta['lora_modules'] == 224,
   f"(e) member {member_root}: {len(hashes)} seed weight files match member.json sha256; base {backbone}")
check = np.load(member_root / 'check.npz', allow_pickle=False)
train_csv = DATA.with_name('train.csv')
train = pd.read_csv(train_csv, dtype=str, keep_default_na=False).set_index('id')
test = pd.read_csv(DATA, dtype={'id': str}, keep_default_na=False)
rows = check['ids'].astype(str)[:24].tolist()
query = pd.concat([pd.DataFrame({'id': rows, 'text': train.loc[rows, 'text'].tolist()}), test[['id', 'text']]], ignore_index=True)
tmp = Path(tempfile.mkdtemp(prefix='ntl-mistral-selftest-'))
query.to_csv(tmp / 'query.csv', index=False)
target = tmp / 'qwen-python'  # the run script installs exactly this wheel set before the Qwen and Mistral workers
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--disable-pip-version-check',
                '--target', str(target), *PATHS['qwen_wheels']], check=True, capture_output=True)
labels = meta['labels']
t0 = time.monotonic()
probabilities, report = member_module.mistral_member(query.id.tolist(), labels, tmp / 'query.csv', tmp, tmp, target,
                                                     deadline=time.monotonic() + 3600)
ok(probabilities is not None and report['used'] and probabilities.shape == (len(query), 18),
   f"(e) mistral_member() ran both seeds in this image on {len(query)} rows (24 training + {len(test)} public test) "
   f"in {time.monotonic() - t0:.0f} s", json.dumps(report, default=str)[-4000:])
expected = []
for seed in meta['seeds']:
    z = check[f'logits_seed{seed}'][:24].astype(np.float64)
    z = np.exp(z - z.max(1, keepdims=True))
    expected.append(z / z.sum(1, keepdims=True))
expected = np.mean(expected, axis=0)
diff = np.abs(probabilities[:24] - expected).max()
agree = float((probabilities[:24].argmax(1) == expected.argmax(1)).mean())
ok(diff <= 0.05 and agree >= 0.95,
   f'(e) this image reproduces the training kernel on the 24 check rows: max |prob diff| {diff:.4f}, argmax agree {agree:.3f}')
timing = [json.loads((tmp / f'mistral_seed{seed}.npz.json').read_text()) for seed in meta['seeds']]
per_row = max(t['predict_seconds'] / t['rows'] for t in timing)
print(f"SELFTEST INFO: Mistral load {max(t['load_seconds'] for t in timing):.0f} s, predict {per_row:.3f} s/row per GPU on "
      f"short batches; peak {max(t['peak_gib'] for t in timing):.1f} GiB", flush=True)
t0 = time.monotonic()
late, late_report = member_module.mistral_member(query.id.tolist(), labels, tmp / 'query.csv', tmp, tmp, target,
                                                 deadline=time.monotonic() + 60)
ok(late is None and late_report['reason'] == 'out of time' and set(late_report['exit_codes'].values()) == {3}
   and time.monotonic() - t0 < 300,
   f"(e) with too little time left mistral_member() returns None (V28's prior kept) in {time.monotonic() - t0:.0f} s")
broken, broken_report = member_module.mistral_member(['x'], labels, tmp / 'missing.csv', tmp, tmp, target)
ok(broken is None and not broken_report['used'], f"(e) a failing member returns None: {broken_report.get('reason')}")
