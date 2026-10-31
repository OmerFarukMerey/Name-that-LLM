# ---- {KERNEL} commit-time self-test (inserted by kaggle/lightning/build_variants.py)
# On the 3-row public test the run cell below skips all inference, so nothing we changed would execute before
# the hidden rerun. This cell checks our edits in both modes, on CPU only, and raises on the first failure.
# Everything lives inside one function: no notebook global is added or shadowed. On any failure the canned
# preview submission is deleted, so a failed commit leaves no submission.parquet behind.
def _ntl_variant_selftest():
    import importlib.util, json, subprocess
    from pathlib import Path
    import numpy as np

    SPEC = None

    def ok(condition, message, detail=''):
        if not condition:
            raise RuntimeError('SELFTEST FAILED: ' + message + ('\n' + detail if detail else ''))
        print('SELFTEST OK: ' + message, flush=True)

    # (a) the recipe the run script reads is the variant's
    recipe = json.loads((RUNTIME / 'recipe.json').read_text(encoding='utf-8'))
    ok(recipe == json.loads(json.dumps(RECIPE)) and recipe['status'] == SPEC['status']
       and recipe['qwen_probability_share'] == SPEC['qwen'] and recipe['helper_probability_share'] == SPEC['helper']
       and recipe['qwen_query_fraction'] == 1.0,
       f"(a) recipe.json: status {recipe['status']}, Qwen {recipe['qwen_probability_share']}, "
       f"helper {recipe['helper_probability_share']}, all rows")

    # (b) the run script has every required line exactly once and none of the replaced ones
    script = (RUNTIME / 'run_fulltext_selected_ensemble.py').read_text(encoding='utf-8')
    counts = {line: script.count(line) for line in SPEC['must']}
    stale = [line for line in SPEC['must_not'] if line in script]
    ok(all(n == 1 for n in counts.values()) and not stale,
       f"(b) run_fulltext_selected_ensemble.py: {len(counts)} required lines once each, {len(SPEC['must_not'])} old ones absent",
       f'counts {counts}\nstale {stale}')

    # (c) the run script's own guard lines accept this recipe, and its blend arithmetic is the variant's
    lines = [line.strip() for line in script.splitlines()]
    guards = [line for line in lines if line.startswith("if recipe.get('status')!=") or line.startswith('if share != ')]
    scope = {'recipe': recipe, 'share': float(recipe['qwen_probability_share']),
             'fraction': float(recipe['qwen_query_fraction'])}
    try:
        for guard in guards:
            exec(guard, {}, scope)
        guard_error = ''
    except ValueError as error:
        guard_error = repr(error)
    ok(len(guards) == 2 and not guard_error, f'(c) both recipe guards of main() pass with this recipe', guard_error)
    rng = np.random.default_rng(0)
    prior = rng.dirichlet(np.ones(18), size=64)
    helper = rng.dirichlet(np.ones(18), size=64)
    mixes = [line for line in lines if line.startswith('prior = ') and line.endswith('*helper')]
    mix_scope = {'prior': prior.copy(), 'helper': helper}
    if len(mixes) == 1:
        exec(mixes[0], {}, mix_scope)
    h = SPEC['helper']
    ok(len(mixes) == 1 and np.allclose(mix_scope['prior'], (1 - h) * prior + h * helper, atol=1e-12, rtol=0),
       f'(c) helper mix line {mixes} equals (1-{h})*prior + {h}*helper')
    module_spec = importlib.util.spec_from_file_location('_ntl_selftest_query_allocation', RUNTIME / 'query_allocation.py')
    allocation = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(allocation)
    labels = [f'label{i}' for i in range(18)]
    ids = [f'row{i}' for i in range(64)]
    logits = rng.normal(size=(64, 18)) * 3
    mixed = allocation.blend_selected(prior, ids, labels, ids, labels, logits, share=scope['share'])
    soft = np.exp(logits - logits.max(1, keepdims=True))
    soft /= soft.sum(1, keepdims=True)
    q = SPEC['qwen']
    ok(np.allclose(mixed, (1 - q) * prior + q * soft, atol=1e-12, rtol=0),
       f"(c) blend_selected(share=recipe Qwen share) equals (1-{q})*prior + {q}*softmax(Qwen logits)")

    # (d) every runtime module still compiles (no file is written)
    modules = sorted(path for path in RUNTIME.rglob('*.py') if path.is_file())
    broken = []
    for path in modules:
        try:
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        except SyntaxError as error:
            broken.append(f'{path}: {error}')
    ok(len(modules) > 10 and not broken, f'(d) all {len(modules)} runtime .py modules compile', '\n'.join(broken))

    # (e) variant-specific checks

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
    _ntl_variant_selftest()
except BaseException:
    OUTPUT.unlink(missing_ok=True)  # the canned 3-row preview written by the first cell
    OUTPUT.with_suffix('.mode.json').unlink(missing_ok=True)
    raise
finally:
    del _ntl_variant_selftest
