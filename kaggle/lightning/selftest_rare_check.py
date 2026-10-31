# ntl-v28-rare-helper15 (selftest_rare_check.py): kirilltym's rare1516 SVC readout replaces V28's ntl627-svc-prefit.
import hashlib, sys
rare_roots = ('/kaggle/input/datasets/kirilltym/ntl-rare-class-svc-1516/rare1516_svc',
              '/kaggle/input/ntl-rare-class-svc-1516/rare1516_svc')
ntl627_root = '/kaggle/input/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets/ntl627-svc-prefit'
rare_files = {  # name: (bytes, sha256), from rare1516_svc/COMPLETE.json of kirilltym/ntl-rare-class-svc-1516 v1
    'COMPLETE.json': (1584, '56a6e180a80cc63234aac7270d4e5fe8ce30a89a9aff8899972e0cefbaba37d3'),
    'readout.npz': (67042500, 'a6516ce884162f9b2dc2af3006c8e71073e990b22c0a66e1b22d5a466970ca79'),
    'small-vocabulary.json': (811146, '7231646eaa7ded3ef0dfca6eacfd54ca2fcf55ff8249e284dc1332286a345595'),
    'shape-idf.npy': (240128, 'ea1bfed92459b51522694e37f36c95434732f132b0cf6cdb6038b578144a3921'),
    'char-vocabulary.json': (2184110, 'bd13435e7f580f79107e9bd7ddfa435b7e9bd495b73af2046ebde6dff0438bc5'),
    'metadata.json': (1596, '1c1b583a89591b34b7c784e994f05afd68f75e95ce4208a1fadb57e1a2aacddd'),
    'word-idf.npy': (720128, 'be6f0b4a6ef03b492263f415d4f8df2ba94488d939422789d679374146c3a511'),
    'char-idf.npy': (1000128, '742a6a60e4e0e7005e5b7cb9f026920e1645f981fa7c108a63e0b9b6abbb9579'),
    'word-vocabulary.json': (2638778, '37ece939e820dec6a4a3ba5e9e78ad238504a0e08c4d17066aed7750cc70c0bb'),
    'shape-vocabulary.json': (568385, '47854fae1d8e2005d1c8fbb7cb46b3ed99dfcc6f3786bf5b59329fb19a80cf71'),
    'small-idf.npy': (200128, 'cee36c1b254bc7428cbddeac6c5dcb6a695b4c4d121de1a9de4c380c53ac2091'),
}
ok(recipe['status'].endswith('_KIRILL15_RARE1516_CANDIDATE') and recipe.get('svc_prefit_dataset') == 'kirilltym/ntl-rare-class-svc-1516/1'
   and script.count("svc_scores=cached_svc(paths['svc_prefit'],texts)") == 1
   and script.count('    total=clipped_logp(svc_scores)\n') == 1 and '*clipped_logp(svc_scores)' not in script,
   f"(e) recipe {recipe['status']} names rare1516; the run script reads paths['svc_prefit'] through cached_svc at weight 1")
root = PATHS.get('svc_prefit')
if not Path('/kaggle/input').exists() and (root is None or not Path(root).exists()):
    print('SELFTEST SKIP: (e) rare1516 inputs: no svc_prefit input and no /kaggle/input (local run); never skipped on Kaggle',
          flush=True)
else:
    root = Path(root)
    paths_json = json.loads((RUNTIME / 'paths.json').read_text(encoding='utf-8'))
    svc_checks = [check for check in REQUIRED_INPUT_CHECKS if check['source'] in ('v17.paths.svc_prefit', 'rare1516.readout')]
    ok(str(root) in rare_roots and [r for r in rare_roots if (Path(r) / 'readout.npz').is_file()] == [str(root)]
       and globals().get('RARE_SVC_ROOT') == paths_json['svc_prefit'] == str(root)
       and 'ntl627-svc-prefit' not in json.dumps(paths_json)
       and svc_checks == [{'source': 'v17.paths.svc_prefit', 'path': str(root)},
                          {'source': 'rare1516.readout', 'path': str(root) + '/readout.npz', 'kind': 'file',
                           'expected_bytes': rare_files['readout.npz'][0]}]
       and not any('ntl627-svc-prefit' in str(check['path']) for check in REQUIRED_INPUT_CHECKS),
       f'(e) paths.json svc_prefit = RARE_SVC_ROOT = {root} (one mount layout); ntl627-svc-prefit unused',
       f"paths.json svc_prefit: {paths_json.get('svc_prefit')}; input checks: {svc_checks}")
    listed = sorted(p.name for p in root.iterdir() if not p.name.startswith('.'))
    mismatched = []
    for name, (size, wanted) in rare_files.items():
        digest = hashlib.sha256()
        if (root / name).is_file():
            with (root / name).open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 22), b''):
                    digest.update(block)
        if not (root / name).is_file() or (root / name).stat().st_size != size or digest.hexdigest() != wanted:
            mismatched.append(f'{name}: {digest.hexdigest() if (root / name).is_file() else "missing"}')
    receipt = json.loads((root / 'COMPLETE.json').read_text(encoding='utf-8')) if not mismatched else {}
    meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8')) if not mismatched else {}
    ok(listed == sorted(rare_files) and not mismatched and receipt.get('status') == 'RARE1516_SVC_COMPLETE'
       and receipt.get('files') == {n: {'bytes': s, 'sha256': h} for n, (s, h) in rare_files.items() if n != 'COMPLETE.json'}
       and meta.get('total_fit_rows') == 79998 and meta.get('extra_rows') == 1516,
       f"(e) rare1516_svc holds exactly its {len(rare_files)} files, each pinned by size and sha256 (readout.npz "
       f"{rare_files['readout.npz'][1][:12]}); COMPLETE.json agrees; 79998 fit rows incl. 1516 extra",
       f'listed {listed}\nmismatched {mismatched}')
    child = RUNTIME.parent / 'ntl_rare_selftest_child.py'
    request = {'rare': str(root), 'ntl627': ntl627_root, 'runtime': str(RUNTIME), 'train_csv': str(DATA.with_name('train.csv'))}
    try:
        run = subprocess.run([sys.executable, '-u', '-B', str(child), json.dumps(request)],
                             capture_output=True, text=True, timeout=1800)
        returncode, stdout, stderr = run.returncode, run.stdout, run.stderr
    except (OSError, subprocess.SubprocessError) as error:
        returncode, stdout, stderr = -1, '', repr(error)
    reports = [line[len('RARE_SELFTEST_REPORT '):] for line in stdout.splitlines() if line.startswith('RARE_SELFTEST_REPORT ')]
    report = json.loads(reports[-1]) if returncode == 0 and len(reports) == 1 else {}
    views = [report.get(name, {}) for name in ('rare', 'ntl627')]
    ok(report.get('module') == str((RUNTIME / 'reference627_common.py').resolve()) and report.get('torch_imported') is False
       and all(v.get('finite') is True and v.get('shape') == [300, 18] for v in views),
       f"(e) child process ran the notebook's cached_svc from {report.get('module')} on rare1516 and ntl627 (finite, no torch)",
       f'returncode {returncode}\nstdout tail:\n{stdout[-3000:]}\nstderr tail:\n{stderr[-3000:]}')
    rare, ref = views
    ok(len(report['labels']) == 18 and rare['labels'] == ref['labels'] == report['labels'] and report['train_labels_known']
       and report['train_rows'] == 300 and report['agreement'] >= 0.90 and rare['accuracy'] >= ref['accuracy'] - 0.02,
       f"(e) on the same 300 train rows (notebook LABELS order): rare1516 vs ntl627 argmax agreement "
       f"{report['agreement']:.4f} (>= 0.90), accuracy {rare['accuracy']:.4f} vs {ref['accuracy']:.4f} (>= ntl627 - 0.02); "
       f"GLM/Kimi hits rare {rare['rare_class_hits']} ntl627 {ref['rare_class_hits']}")
