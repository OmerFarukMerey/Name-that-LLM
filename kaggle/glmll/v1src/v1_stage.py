"""V1 stage of the submission notebook (docs/12_v1_verosimilitud_glm.md, fase 2.3).

Runs after S10's ensemble.py has written the submission and its final log-probabilities (final-logits.npz).
1. Partial coverage (doc 12, 2.5b): the whole test does not fit after S10 (~6.7 h on 4,000 texts), and nearly all
   the gain of the member is telling glm-5.3 from glm-5.3-flash. So the test texts are ranked by the S10
   probability of those two classes (--select-classes) and the top ones are scored: as many as the time left
   allows, at most --max-cover of the test (train OOF: 15 % of the texts give +0.030, 20 % +0.0315, all +0.029).
   If fewer than --min-texts fit, stops (the S10 submission is kept as it is).
2. Scores those texts with run_pipeline.py (ntl-segrun), each model with a hard time budget; if a budget is not
   enough (exit code 3), stops the same way. With --prefetch the pipeline reads the next segment while the current
   one computes (and the first GLM-5.3 segment during the last Flash one). If Flash took longer than estimated,
   GLM-5.3 scores only the top of the ranking that still fits (at least --min-texts) instead of giving up, and the
   blend uses those texts.
3. Likelihood member: logistic regression exported by src/fork_v1/export_ll_model.py (plain numpy here).
4. Blend: the S10 final log-probs z are rescaled to the entropy of the bank base the weight was chosen against
   (temperature T', bisection, over the whole test), and on the scored texts argmax(z + (W / T') * logp_member)
   replaces the S10 prediction; the other texts keep it.

Exit code 0 both when the blend is applied and when it falls back to S10 for lack of time (the report says
which); anything else is a bug and the notebook raises, so a broken V1 shows up as a failed submission and not as
an S10 score. That includes the sanity guard: on train OOF the blend changes ~37% of the scored texts (~11% of
the test at 30% coverage); if it changes more than --max-changed (25%) of the whole test, the features are probably
unlike train and nothing is submitted.

    python v1_stage.py --test-csv T --final-logits F --out submission.parquet --report report.json \\
        --bin BIN --pipeline run_pipeline.py --flash-segs 'G1' --glm53-segs 'G2' --model ll_model.json \\
        --work W --outdir O --deadline-mono D [--seq-batch 8] [--pack --ubatch-div 2: NOT exact on GPU, see doc 12] \\
        [--fixed-s-flash A --row-s-flash B --fixed-s-glm53 C --row-s-glm53 D]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

PREFIX = '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>'


def entropy(z):
    return float(-(np.exp(z) * z).sum(1).mean())


def match_temperature(z, h_ref):
    """T' with mean entropy of log_softmax(T' z) equal to h_ref (entropy decreases with T')."""
    lo, hi = 1e-3, 1e3
    if entropy(log_softmax(z * hi, 1)) > h_ref or entropy(log_softmax(z * lo, 1)) < h_ref:
        raise ValueError('entropy_ref out of reach')
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if entropy(log_softmax(z * mid, 1)) > h_ref:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def member_logp(model, feats, ids):
    """feats: {'flash': DataFrame, 'glm53': DataFrame} indexed by id -> (n, 18) log-probs, model['classes'] order."""
    blocks = [feats[t].add_prefix(f'{t}:') for t in ('flash', 'glm53')]
    F = pd.concat(blocks, axis=1).reindex(ids)
    for c in model['diff_features']:
        F[f'd:{c}'] = F[f'flash:{c}'] - F[f'glm53:{c}']
    X = F[model['features']].to_numpy(np.float64)
    assert np.isfinite(X).all(), 'non-finite likelihood features'
    z = ((X - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']).T + np.asarray(model['intercept'])
    return log_softmax(z, 1)


def main():
    ap = argparse.ArgumentParser()
    for k in ('--test-csv', '--final-logits', '--out', '--report', '--bin', '--pipeline', '--flash-segs', '--glm53-segs',
              '--model', '--work', '--outdir'):
        ap.add_argument(k, required=True)
    ap.add_argument('--deadline-mono', type=float, required=True, help='time.monotonic() by which all must be done')
    ap.add_argument('--pack', action='store_true')
    ap.add_argument('--ubatch-div', type=int, default=1)
    ap.add_argument('--seq-batch', type=int, default=8)
    ap.add_argument('--max-text-tokens', type=int, default=0,
                    help='0 = the cut the member was trained with (model max_text_tokens); any other value must match it')
    ap.add_argument('--fixed-s-flash', type=float, default=870.0, help='time estimate: per call (loads)')
    ap.add_argument('--row-s-flash', type=float, default=0.0141, help='time estimate: per token row')
    ap.add_argument('--fixed-s-glm53', type=float, default=1140.0)
    ap.add_argument('--row-s-glm53', type=float, default=0.0231)
    ap.add_argument('--select-classes', default='glm-5.3,glm-5.3-flash',
                    help='texts are scored in decreasing order of the S10 probability of these classes')
    ap.add_argument('--max-cover', type=float, default=0.30, help='at most this fraction of the test is scored')
    ap.add_argument('--min-texts', type=int, default=50, help='fewer texts than this fit: keep S10')
    ap.add_argument('--reserve-s', type=float, default=180.0, help='kept for the blend and the writer')
    ap.add_argument('--prefetch', action='store_true', help='run_pipeline --prefetch (page-cache warm-up of segments)')
    ap.add_argument('--max-changed', type=float, default=0.25,
                    help='sanity guard: the blend changes ~11%% of the predictions on train OOF (W=1); far more means '
                         'something is broken (e.g. features unlike train): fail instead of submitting it')
    a = ap.parse_args()

    t0 = time.monotonic()
    rep = {'status': 'started', 'deadline_left_s': a.deadline_mono - t0}
    out_report = Path(a.report)

    def save(**kw):
        rep.update(kw)
        out_report.write_text(json.dumps(rep, indent=1))
        print('V1', json.dumps({k: v for k, v in kw.items() if not isinstance(v, dict)}), flush=True)

    def left():
        return a.deadline_mono - time.monotonic() - a.reserve_s

    model = json.loads(Path(a.model).read_text())
    assert model['format'] == 'ntl-v1-ll-model-1'
    a.max_text_tokens = a.max_text_tokens or model['max_text_tokens']
    assert a.max_text_tokens == model['max_text_tokens'], 'the test must be cut like the train features'
    frame = pd.read_csv(a.test_csv, dtype={'id': str})
    frame['text'] = frame.text.fillna('').astype(str)
    ids = frame.id.tolist()
    with np.load(a.final_logits, allow_pickle=False) as f:
        assert f['ids'].astype(str).tolist() == ids, 'final-logits ids differ from the test order'
        labels = f['labels'].astype(str).tolist()
        z = f['logits'].astype(np.float64)
    assert sorted(labels) == sorted(model['classes']) and z.shape == (len(ids), 18) and np.isfinite(z).all()

    zl = log_softmax(z, 1)
    sel_cls = a.select_classes.split(',')
    assert all(c in labels for c in sel_cls), sel_cls
    mass = np.exp(zl[:, [labels.index(c) for c in sel_cls]]).sum(1)
    order = np.argsort(-mass, kind='stable')

    # time estimate from the characters (tokens per character of train, cut at max-text-tokens, + prefix): the
    # largest number of top-ranked texts whose two passes fit in the time left with a 10 % margin
    tpc = model.get('tok_per_char', 0.235)
    rows_i = np.minimum(frame.text.str.len().to_numpy() * tpc * 1.05, a.max_text_tokens) + 13
    cum = np.cumsum(rows_i[order])
    est = lambda r: (a.fixed_s_flash + a.row_s_flash * r, a.fixed_s_glm53 + a.row_s_glm53 * r)
    fits = 1.1 * (a.fixed_s_flash + a.fixed_s_glm53 + (a.row_s_flash + a.row_s_glm53) * cum) <= left()
    n_sel = int(min(fits.sum(), int(a.max_cover * len(ids))))   # fits is monotone: its count is the largest prefix
    rows = int(cum[n_sel - 1]) if n_sel else 0
    est_f, est_g = est(rows)
    save(status='estimated', n_texts=len(ids), n_selected=n_sel, cover=n_sel / len(ids),
         mass_threshold=float(mass[order[n_sel - 1]]) if n_sel else None, est_rows=rows,
         est_rows_all=int(cum[-1]), est_flash_s=est_f, est_glm53_s=est_g, left_s=left())
    if n_sel < a.min_texts:
        save(status='fallback_s10', reason=f'only {n_sel} texts fit in the time left')
        return
    sel = np.sort(order[:n_sel])
    sel_ids = [ids[i] for i in sel]

    Path(a.work).mkdir(parents=True, exist_ok=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    times = {}
    # Flash keeps back only what GLM-5.3 needs for the minimum (it can shrink to the top texts after Flash)
    reserve_flash = 1.05 * (a.fixed_s_glm53 + a.row_s_glm53 * cum[min(a.min_texts, n_sel) - 1])
    for tag, segs, reserve in (('flash', a.flash_segs, reserve_flash), ('glm53', a.glm53_segs, 0.0)):
        if tag == 'glm53':
            # the largest top prefix (of the texts Flash scored) whose GLM-5.3 pass still fits, with a 10 % margin
            m = int(min(n_sel, (1.1 * (a.fixed_s_glm53 + a.row_s_glm53 * cum[:n_sel]) <= left()).sum()))
            if m < a.min_texts:
                save(status='fallback_s10', reason=f'after Flash only {m} texts fit for GLM-5.3', times=times)
                return
            if m < n_sel:
                save(status='glm53_shrunk', n_selected_flash=n_sel, n_selected=m, cover=m / len(ids),
                     mass_threshold=float(mass[order[m - 1]]), left_s=left())
                n_sel = m
                sel = np.sort(order[:n_sel])
                sel_ids = [ids[i] for i in sel]
        budget = left() - reserve
        if budget <= 0:
            save(status='fallback_s10', reason=f'no time left for {tag}')
            return
        query = Path(a.work) / f'v1_query_{tag}.csv'
        frame.iloc[sel][['id', 'text']].to_csv(query, index=False)
        cmd = [sys.executable, a.pipeline, '--bin', a.bin, '--segs', segs, '--csv', str(query), '--out', a.outdir,
               '--tag', tag, '--work', str(Path(a.work) / 'state'), '--seq-batch', str(a.seq_batch),
               '--state-dtype', 'auto', '--max-text-tokens', str(a.max_text_tokens), '--prefix', PREFIX,
               '--budget-s', f'{budget:.0f}']
        if a.pack:
            cmd.append('--pack')
        if a.ubatch_div > 1:
            cmd += ['--ubatch-div', str(a.ubatch_div)]
        if a.prefetch:
            cmd.append('--prefetch')
            if tag == 'flash':
                cmd += ['--prefetch-after', sorted(glob.glob(a.glm53_segs))[0]]
        t = time.monotonic()
        with open(Path(a.outdir) / f'{tag}_pipeline.txt', 'w') as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        times[tag] = time.monotonic() - t
        if code == 3:
            save(status='fallback_s10', reason=f'{tag} time budget exceeded', times=times)
            return
        if code != 0:
            raise RuntimeError(f'V1 {tag} pipeline failed with exit code {code}; see {tag}_pipeline.txt')
        save(status=f'{tag}_done', times=times, left_s=left())

    feats = {t: pd.read_csv(Path(a.outdir) / f'{t}_features.csv', dtype={'id': str}).set_index('id')
             for t in ('flash', 'glm53')}
    for t, f in feats.items():
        assert set(sel_ids) <= set(f.index), f'{t}: features missing for some texts'
        feats[t] = f.loc[sel_ids]
    ll = member_logp(model, feats, sel_ids)[:, [model['classes'].index(c) for c in labels]]

    tp = match_temperature(zl, model['entropy_ref'])
    w = model['W'] / tp
    before = zl.argmax(1)
    after = before.copy()
    after[sel] = (zl[sel] + w * ll).argmax(1)
    changed = float((before != after).mean())
    changed_sel = float((before[sel] != after[sel]).mean())
    agrees = float((ll.argmax(1) == before[sel]).mean())
    if changed > a.max_changed:
        save(status='error_sanity', changed=changed, changed_selected=changed_sel, temperature=tp, weight=w,
             member_argmax_agrees_s10=agrees)
        raise RuntimeError(f'V1 blend changes {changed:.1%} of the predictions (> {a.max_changed:.0%}): not submitted')
    sub = pd.DataFrame({'id': ids, 'model': np.asarray(labels)[after]})
    assert len(sub) == len(frame) and sub.id.is_unique and sub.model.isin(labels).all()
    sub.to_parquet(a.out, index=False)
    assert pd.read_parquet(a.out).equals(sub)
    count = lambda p: {labels[i]: int(n) for i, n in zip(*np.unique(p, return_counts=True))}
    save(status='applied', n_blended=len(sel_ids), entropy_s10=entropy(zl), temperature=tp, weight=w, changed=changed,
         changed_selected=changed_sel, member_argmax_agrees_s10=agrees,
         counts_s10=count(before), counts_v1=count(after), times=times, total_s=time.monotonic() - t0)


if __name__ == '__main__':
    main()
