"""ntl-v28-mistral-*: the full-fit Mistral-7B member (omerfarukmerey/ntl-member-mistral7b-full) for the run script.

mistral_member() runs mistral_infer.py (written next to this module) once per trained seed, one GPU each, and returns
the seeds' mean probabilities in the caller's row and label order. It never raises. On any failure, or when a seed
cannot finish before DEADLINE_HOURS after the notebook started (mistral_infer.py then exits 3 without output), it
returns None and the run script keeps V28's prior unchanged. The report says which case happened.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MEMBER_ROOTS = ['/kaggle/input/notebooks/omerfarukmerey/ntl-member-mistral7b-full/mistral_full',
                '/kaggle/input/ntl-member-mistral7b-full/mistral_full']
BASE_ROOT = '/kaggle/input/models/mistral-ai/mistral'
DEADLINE_HOURS = 8.5  # Kaggle stops the hidden rerun at 9 h; everything after this member takes seconds
OUT_OF_TIME = 3


def resolve():
    roots = [root for root in MEMBER_ROOTS if (Path(root) / 'member.json').is_file()]
    if len(roots) != 1:
        raise FileNotFoundError(f'Mistral member roots found: {roots}')
    configs = glob.glob(BASE_ROOT + '/**/config.json', recursive=True)
    if len(configs) != 1:
        raise FileNotFoundError(f'Mistral base configs found: {configs}')
    return Path(roots[0]), str(Path(configs[0]).parent)


def mistral_member(ids, labels, query_path, work, logs, pythonpath, deadline=None):
    """(mean probabilities [len(ids), 18] or None, report)."""
    started = time.monotonic()
    report = dict(event='MISTRAL_MEMBER', used=False)
    procs = []
    try:
        member, backbone = resolve()
        meta = json.loads((member / 'member.json').read_text(encoding='utf-8'))
        if sorted(meta['labels']) != sorted(labels) or len(set(labels)) != 18:
            raise ValueError('Mistral member labels differ')
        if deadline is None:
            deadline = float(os.environ['NTL_STARTED_MONOTONIC']) + DEADLINE_HOURS * 3600
        report['minutes_left_at_start'] = round((deadline - started) / 60, 1)
        infer = Path(__file__).resolve().parent / 'mistral_infer.py'
        env = dict(os.environ)
        if pythonpath:
            env['PYTHONPATH'] = str(pythonpath)
        for gpu, seed in enumerate(meta['seeds']):
            out = Path(work) / f'mistral_seed{seed}.npz'
            out.unlink(missing_ok=True)
            log_path = Path(logs) / f'mistral_seed{seed}.log'
            command = [sys.executable, '-u', str(infer), '--backbone', backbone, '--member', str(member),
                       '--seed', str(seed), '--query', str(query_path), '--out', str(out), '--deadline', repr(deadline),
                       '--start-delay', str(90 * gpu)]
            with log_path.open('w', encoding='utf-8') as log:
                procs.append((seed, out, log_path, subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))))
        codes = {seed: proc.wait() for seed, _, _, proc in procs}
        report['exit_codes'] = codes
        if any(codes.values()):
            report['reason'] = 'out of time' if OUT_OF_TIME in codes.values() else 'worker failed'
            report['log_tails'] = {seed: path.read_text(errors='replace')[-1500:] for seed, _, path, _ in procs}
            return None, report
        lookup = {name: i for i, name in enumerate(ids)}
        if len(lookup) != len(ids):
            raise ValueError('Duplicate query IDs')
        probabilities = []
        for seed, out, _, _ in procs:
            with np.load(out, allow_pickle=False) as z:
                saved, names = z['ids'].astype(str).tolist(), z['labels'].astype(str).tolist()
                if len(saved) != len(ids) or set(saved) != set(lookup) or sorted(names) != sorted(labels):
                    raise ValueError(f'Mistral seed {seed}: rows or labels differ')
                logits = np.zeros((len(ids), len(labels)), np.float64)
                logits[[lookup[i] for i in saved]] = z['logits'][:, [names.index(l) for l in labels]]
            if not np.isfinite(logits).all():
                raise ValueError(f'Mistral seed {seed}: non-finite logits')
            p = np.exp(logits - logits.max(1, keepdims=True))
            probabilities.append(p / p.sum(1, keepdims=True))
        mean = np.mean(probabilities, axis=0)
        if not (np.isfinite(mean).all() and np.allclose(mean.sum(1), 1, atol=1e-6)):
            raise ValueError('Invalid Mistral probabilities')
        report.update(used=True, seeds=meta['seeds'],
                      seed_argmax_agree=float((probabilities[0].argmax(1) == probabilities[-1].argmax(1)).mean()))
        return mean, report
    except Exception as error:  # V28 must still finish: the caller keeps its prior
        report['reason'] = repr(error)
        return None, report
    finally:
        for _, _, _, proc in procs:
            if proc.poll() is None:
                proc.kill()
        report['seconds'] = round(time.monotonic() - started, 1)
