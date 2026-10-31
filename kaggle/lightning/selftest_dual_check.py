import sys
worker = (RUNTIME / 'own_qwen_worker.py').read_text(encoding='utf-8')
counts = {line: worker.count(line) for line in SPEC['worker_must']}
ok(all(n == 1 for n in counts.values()) and 'logits=model.predict(texts,batch_size=3)\n' not in worker
   and worker.index('from dual_head import') < worker.index('def main():')
   and worker.index('smoke=smoke_test()') < worker.index('groups,valid=dual_groups(texts)') < worker.index('model=FullTextClassifier('),
   f'(e) own_qwen_worker.py: {len(counts)} dual-head lines once each; dual_head imported at the top, its smoke test '
   'and the real-text clustering run before the 27B model loads', f'counts {counts}')
train_csv = DATA.with_name('train.csv')
if not train_csv.is_file():
    ok(not Path('/kaggle/input').exists(), f'(e) {train_csv} is absent only because this is a local run (no /kaggle/input)')
    print('SELFTEST SKIP: (e) no train.csv: the dual-head child uses its synthetic fixture instead of train passages', flush=True)
child = RUNTIME.parent / Path(SPEC['child']).name
request = {'runtime': str(RUNTIME), 'train_csv': str(train_csv) if train_csv.is_file() else None}
try:
    run = subprocess.run([sys.executable, '-u', '-B', str(child), json.dumps(request)],
                         capture_output=True, text=True, timeout=1800)
    returncode, stdout, stderr = run.returncode, run.stdout, run.stderr
except (OSError, subprocess.SubprocessError) as error:
    returncode, stdout, stderr = -1, '', repr(error)
reports = [line[len('DUAL_SELFTEST_REPORT '):] for line in stdout.splitlines() if line.startswith('DUAL_SELFTEST_REPORT ')]
report = json.loads(reports[-1]) if returncode == 0 and len(reports) == 1 else {}
expected_modules = {name: str((RUNTIME / (name + '.py')).resolve())
                    for name in ('dual_head', 'own_qwen_worker', 'run_fulltext_selected_ensemble')}
ok(report.get('modules') == expected_modules,
   f"(e) dual-head child ran the notebook's dual_head, own_qwen_worker and run script from RUNTIME ({report.get('fixture')} fixture)",
   f'returncode {returncode}\nstdout tail:\n{stdout[-3000:]}\nstderr tail:\n{stderr[-3000:]}')
smoke = report['smoke']
ok(smoke['qwen_dual_valid_rows'] == 10 and smoke['smoke_rows'] == 40,
   f"(e) dual_head.smoke_test() (the worker's pre-load check) finds its planted 10-row cluster among {smoke['smoke_rows']} rows")
ok(report['planted_group_exact'] and report['fixture_valid_rows'] >= 10,
   f"(e) topic_groups(.90) puts exactly the 10 planted near-duplicates in one valid cluster "
   f"({report['fixture_valid_rows']} valid rows of 60 in the {report['fixture']} fixture)")
ok(report['predict_calls'] == [[60, 3, True]] and report['npz_keys'] == ['centered_groups', 'centered_logits', 'centered_valid', 'ids', 'labels', 'logits']
   and report['npz_dtypes']['logits'] == report['npz_dtypes']['centered_logits'] == 'float32'
   and report['npz_dtypes']['centered_valid'] == 'uint8' and report['worker_valid_matches'] and report['sidecar_matches'],
   f"(e) patched worker main() with a mock 27B: predict(return_features=True) once, qwen.npz keys/dtypes, valid rows and "
   f"sidecar (centered_valid_rows {report['sidecar_dual']['centered_valid_rows']}, histogram "
   f"{report['sidecar_dual']['cluster_size_histogram']}) match an independent recomputation")
ok(report['centered_max_abs_diff'] <= 2e-2 and report['outside_rows_identical'],
   f"(e) worker centred logits equal float64 cluster-mean centring + model.centered (max |diff| {report['centered_max_abs_diff']:.1e}); "
   'other rows keep the plain logits bit for bit')
ok(report['block_max_abs_diff'] <= 1e-12 and report['block_outside_bitwise_v28'] and report['block_inside_changed']
   and report['blend_npz_matches'] and report['dual_report_matches'] and report['block_lines'] >= 10,
   f"(e) patched run-script block via aligned(..., field='centered_logits'): .9*prior+.1*dual (Roman's alpha .35/.18) on "
   f"valid rows (max |diff| {report['block_max_abs_diff']:.1e}), V28's mix bit for bit elsewhere; report {report['dual_report']}")
ok(report['formula_max_abs_diff'] <= 1e-12 and min(report['formula_branches']) >= 50 and report['formula_invalid_identity'],
   f"(e) dual formula equals Roman's on 400 random rows ({report['formula_branches'][0]} agree, "
   f"{report['formula_branches'][1]} disagree); with no valid row the plain logits pass through unchanged")
ok(report['fail_closed'] == {'centred_differs_outside': 'ValueError', 'features_misaligned': 'FloatingPointError',
                             'partial_cluster': 'ValueError', 'valid_not_binary': 'ValueError'},
   f"(e) dual head fails closed on a broken contract: {report['fail_closed']}")
