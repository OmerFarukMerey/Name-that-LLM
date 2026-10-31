# ---- ntl-0641-svc-tiers commit-time self-test (inserted by kaggle/lightning/build.py)
# On the 3-row public test the run cell below skips all inference, so nothing we changed would execute
# before the hidden rerun. This cell checks our changes in both modes, on CPU only, and raises on the
# first failure. The SVC call runs in a child process (the script written by the cell above), so this
# process never imports sklearn or keeps the readout. Everything lives inside one function: no notebook
# global is added or shadowed. On any failure the canned preview submission is deleted, so a failed
# commit leaves no submission.parquet behind.
def _ntl_svc_tiers_selftest():
    import hashlib, json, subprocess, sys
    from pathlib import Path

    def ok(condition, message, detail=''):
        if not condition:
            raise RuntimeError('SELFTEST FAILED: ' + message + ('\n' + detail if detail else ''))
        print('SELFTEST OK: ' + message, flush=True)

    roots = ('/kaggle/input/datasets/omerfarukmerey/ntl-svc627-quality-tiers', '/kaggle/input/ntl-svc627-quality-tiers')
    root = Path(SVC_TIERS_ROOT)
    child = Path('/kaggle/temp/ntl_svc_tiers_selftest_child.py')

    # (a) the ensemble reads our refit through paths.json; ntl627-svc-prefit is not used anywhere
    paths_text = (RUNTIME / 'paths.json').read_text(encoding='utf-8')
    svc_checks = [check for check in REQUIRED_INPUT_CHECKS if check['source'] == 'v17.paths.svc_prefit']
    ok(json.loads(paths_text)['svc_prefit'] == PATHS['svc_prefit'] == str(root) and str(root) in roots
       and (root / 'readout.npz').is_file() and 'ntl627-svc-prefit' not in paths_text
       and [check['path'] for check in svc_checks] == [str(root)]
       and not any('ntl627-svc-prefit' in str(check['path']) for check in REQUIRED_INPUT_CHECKS),
       f'(a) paths.json svc_prefit = {root}; ntl627-svc-prefit unused',
       f"paths.json svc_prefit: {json.loads(paths_text).get('svc_prefit')}; input check: {svc_checks}")

    # (b) the SVC log-probability enters the blend x1.64, exactly once; the recipe guards are untouched
    script = (RUNTIME / 'run_fulltext_selected_ensemble.py').read_text(encoding='utf-8')
    ok(script.count('total=1.64*clipped_logp(svc_scores)') == 1 and 'total=clipped_logp(svc_scores)' not in script
       and script.count("svc_scores=cached_svc(paths['svc_prefit'],texts)") == 1
       and "recipe.get('status')!='FROZEN_ALLROWS_QWEN05_KIRILL10_CANDIDATE'" in script and 'share != .05' in script,
       '(b) run_fulltext_selected_ensemble.py has total=1.64*clipped_logp(svc_scores) once; Qwen5%/Kirill10% guards intact',
       '\n'.join(line for line in script.splitlines() if 'svc_scores' in line or 'share' in line))

    # (c) every file of the mounted dataset is the version we uploaded (unscaled tiered refit, same vocab/idf)
    expected = {
        'COMPLETE.json': '7b020aecf594bb965e252ae0bb9e70a90859430534065401c43775950cfd92c8',
        'README.md': 'd02c0ca519a7240b04fff3ca533694b05e00f3e3ed6954f7d09c515c1f4c7a58',
        'char-idf.npy': '338579f72895d83add4879020c61f8eba11162b49d25a47d04cbbd4edeb7d4d0',
        'char-vocabulary.json': 'cf87df15e4da885cd6ae7db74d58434a98a38a1a9637096f35809e757cab4c93',
        'metadata.json': '2f730ffa124cb02389793d4f6371d17839cb0095340f7727d425000a79adc5b3',
        'preview.npz': '1a1c939cf6068d78aa7e2b97c325af875bbf09b362d40a622fc713fcbfdd483e',
        'publication-manifest.json': 'c79cd56d2472e926d2eda49ee097ff027b8f1b9edcc7d470ffae4b63aa4895b5',
        'readout.npz': '5827d9dac1dceef16b43c9d7d4620c067627997b3046730ff61e54ec6f205aa0',
        'shape-idf.npy': '7b0ec06d411157eea3fa2a7f26ef2ad84420e0c02b88e7a6fbdbca83aa30f1c3',
        'shape-vocabulary.json': 'd61b7ca89fff0b357752bf47d8e7bb857e826b3f3436330c373d980ed594ba41',
        'small-idf.npy': 'daba3c302dedc9bcc0e3db0dae6b88df1787131ddd993ecc4fd8c0048fdfdf32',
        'small-vocabulary.json': 'fe7d0965ecf6fa854fe5661296f6c7389008a4ac861787b6db19e5e871d722ae',
        'word-idf.npy': 'bbef54e4e8ea40dbb1f063b504317e09121c42481205827cfd9b3bc8c4d2a9f1',
        'word-vocabulary.json': '1f2e9c31751129e8ca072595b3ddea777fc95c467428c4777dfc778c455fe927',
    }
    mismatched = []
    for name, wanted in expected.items():
        digest = hashlib.sha256()
        if (root / name).is_file():
            with (root / name).open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 22), b''):
                    digest.update(block)
        if not (root / name).is_file() or digest.hexdigest() != wanted:
            mismatched.append(f'{name}: {digest.hexdigest() if (root / name).is_file() else "missing"}')
    ok(not mismatched, f'(c) sha256 of all {len(expected)} dataset files match the upload (readout.npz 5827d9dac1dc)',
       '\n'.join(mismatched))

    # (d)+(e) one CPU call of the notebook's own cached_svc in a child process: public test rows, then 300 train rows
    fixture = {record['id']: record['fingerprint'] for record in PUBLIC_FIXTURE['records']}
    request = {'root': str(root), 'runtime': str(RUNTIME), 'test_csv': str(DATA),
               'train_csv': str(DATA.with_name('train.csv')), 'fixture': fixture}
    try:
        run = subprocess.run([sys.executable, '-u', '-B', str(child), json.dumps(request)],
                             capture_output=True, text=True, timeout=1800)
        returncode, stdout, stderr = run.returncode, run.stdout, run.stderr
    except (OSError, subprocess.SubprocessError) as error:
        returncode, stdout, stderr = -1, '', repr(error)
    reports = [line[len('SVC_SELFTEST_REPORT '):] for line in stdout.splitlines() if line.startswith('SVC_SELFTEST_REPORT ')]
    report = json.loads(reports[-1]) if returncode == 0 and len(reports) == 1 else {}
    ok(report.get('module') == str((RUNTIME / 'reference627_common.py').resolve()) and report.get('finite') is True
       and report.get('torch_imported') is False and report.get('logits_shape') == [len(report['public_ids']) + 300, 18],
       f"(d) child process ran the notebook's cached_svc from {report.get('module')} (finite logits, no torch)",
       f'returncode {returncode}\nstdout tail:\n{stdout[-3000:]}\nstderr tail:\n{stderr[-3000:]}')
    public = report['public_ids']
    ok(sorted(report['preview_ids']) == sorted(fixture) and report['direct_shape'] == [3, 18] and set(public) <= set(fixture)
       and (len(public) == 3 or not preview),
       f'(d) preview.npz covers the 3 public ids; {len(public)} of them are in this test.csv')
    if public:
        gap = float('nan') if report['gap'] is None else report['gap']
        ok(gap <= 1e-6, f'(d) cached_svc on {len(public)} public test rows equals preview.npz direct (max |diff| {gap:.1e})')
    else:
        print('SELFTEST SKIP: (d) no public row in this test.csv (hidden rerun); it was checked at commit', flush=True)
    ok(len(report['labels']) == 18 and report['train_labels_known'] and report['train_rows'] == 300
       and report['accuracy'] >= 0.95,
       f"(e) cached_svc argmax in notebook LABELS order matches {report['accuracy']:.4f} of {report['train_rows']} train rows (>= 0.95)")

    # (f) every attached input, exactly as the run cell checks them (it only does so on the hidden rerun)
    missing = []
    for check in REQUIRED_INPUT_CHECKS:
        location = Path(check['path'])
        okay = location.exists()
        if check.get('kind') == 'file':
            okay = okay and location.is_file()
        if check.get('kind') == 'directory':
            okay = okay and location.is_dir()
        if okay and 'expected_bytes' in check:
            okay = location.stat().st_size == check['expected_bytes']
        if not okay:
            missing.append(check['source'] + ': ' + str(location))
    ok(not missing and PATHS['test_csv'] == str(DATA) and PATHS['own_fulltext_adapter'] == str(FINAL_ADAPTER),
       f'(f) all {len(REQUIRED_INPUT_CHECKS)} REQUIRED_INPUT_CHECKS pass (exists, kind, expected_bytes)',
       'Required attached input mismatch:\n' + '\n'.join(missing))

    # (g) two GPUs, counted without initialising CUDA in this process
    try:
        listing = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=120)
        returncode, listed = listing.returncode, listing.stdout + listing.stderr
    except (OSError, subprocess.SubprocessError) as error:
        returncode, listed = -1, repr(error)
    gpus = [line for line in listed.splitlines() if line.startswith('GPU ')]
    ok(returncode == 0 and len(gpus) == 2, f'(g) nvidia-smi -L lists {len(gpus)} GPUs', listed)

    # (h) no legacy short mount path survived the rewrite in any written module, ledger or check
    legacy = ('/kaggle/input/ntl-olmo2-native24-final-inference-assets', '/kaggle/input/nlm-l24-adapters')
    written = [p for p in RUNTIME.rglob('*') if p.suffix in ('.py', '.json') and p.is_file() and not p.is_symlink()]
    stale = [str(p) for p in written if any(s in p.read_text(encoding='utf-8', errors='replace') for s in legacy)]
    stale += [check['source'] for check in REQUIRED_INPUT_CHECKS if str(check['path']).startswith(legacy)]
    ok(not stale, f'(h) no legacy short mount path in {len(written)} runtime files or REQUIRED_INPUT_CHECKS',
       '\n'.join(stale))

    print('SELFTEST ALL PASSED', flush=True)


try:
    _ntl_svc_tiers_selftest()
except BaseException:
    OUTPUT.unlink(missing_ok=True)  # the canned 3-row preview written by the first cell
    OUTPUT.with_suffix('.mode.json').unlink(missing_ok=True)
    raise
finally:
    del _ntl_svc_tiers_selftest
