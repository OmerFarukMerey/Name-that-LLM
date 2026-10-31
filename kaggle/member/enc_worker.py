"""Two-stage full fine-tune of one encoder: stage 1 on a public pool, stage 2 on the official rows.

    python enc_worker.py --model FacebookAI/roberta-large --pool pool.parquet --train train.parquet --labels '[...]'
                         --out DIR [--max-len 384] [--batch 16] [--accum 1] [--lr1 2e-5] [--epochs1 2]
                         [--lr2 1e-5] [--epochs2 3] [--stage1-hours 4.5] [--seed 0]

The recipe Dhruv Tibarewal gives for his ModernBERT-large member (dataset nlm-enc-v5, notebook "Name That LLM Fully
Reproducible"): full fine-tune, mean of the last layer, a linear head, two epochs on the pool at 2e-5 (class-weighted
here, as his LoRA stage 1), then three on the official rows at 1e-5. Pooling is the mean of the last hidden state over
every token the attention mask keeps, special tokens included, which is how nlm-v27's infer.py reads an 'enc' member.

Stage 2 runs once per outer fold of train.parquet's `fold` column (out-of-fold logits for that fold's rows) and once
on every row; only the all-rows model is saved. Output in nlm-enc-v5's layout, so infer.py's 'enc' type loads it
unchanged: DIR/encoder/ (save_pretrained, float32, with the tokenizer), DIR/head.safetensors (weight [18, hidden],
bias [18]), DIR/meta.json (labels, max_len, pool, model), plus DIR/oof.npz (ids, fold, logits) and DIR/record.json.
--stage1-hours stops stage 1 early (with a log line) so the kernel stays inside Kaggle's session limit.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

T0 = time.time()


def log(*x):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *x, flush=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr1", type=float, default=2e-5)
    p.add_argument("--epochs1", type=float, default=2)
    p.add_argument("--lr2", type=float, default=1e-5)
    p.add_argument("--epochs2", type=float, default=3)
    p.add_argument("--head-lr-mult", type=float, default=20)
    p.add_argument("--stage1-hours", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="local smoke test: CPU, no autocast")
    return p.parse_args()


class Encoder(torch.nn.Module):
    def __init__(self, name, n_labels):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, n_labels)

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))


def batches(lengths, batch, rng, shuffle=True):
    """Length-grouped batches: shuffle, sort within chunks of 50 batches, shuffle the batch order."""
    idx = rng.permutation(len(lengths)) if shuffle else np.argsort(lengths, kind="stable")
    if not shuffle:
        return [idx[i:i + batch] for i in range(0, len(idx), batch)]
    out = []
    for s in range(0, len(idx), batch * 50):
        chunk = idx[s:s + batch * 50]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        out += [chunk[i:i + batch] for i in range(0, len(chunk), batch)]
    return [out[i] for i in rng.permutation(len(out))]


def collate(enc, b, pad, device):
    width = max(len(enc[i]) for i in b)
    ids = torch.full((len(b), width), pad, dtype=torch.long)
    mask = torch.zeros((len(b), width), dtype=torch.long)
    for r, i in enumerate(b):
        ids[r, :len(enc[i])] = torch.tensor(enc[i])
        mask[r, :len(enc[i])] = 1
    return ids.to(device), mask.to(device)


def train(model, enc, y, weights, args, lr, epochs, rng, device, pad, hours=None, tag=""):
    """AdamW, linear warm-up (6%) and decay, fp16 autocast with a grad scaler, gradient clipping 1.0."""
    lengths = np.array([len(x) for x in enc])
    steps_per_epoch = math.ceil(len(enc) / args.batch / args.accum)
    total = max(1, int(steps_per_epoch * epochs))
    groups = [{"params": model.encoder.parameters(), "lr": lr},
              {"params": model.head.parameters(), "lr": lr * args.head_lr_mult}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01, eps=1e-6)
    base = [g["lr"] for g in opt.param_groups]
    warm = max(1, int(0.06 * total))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    yt = torch.as_tensor(y, dtype=torch.long)
    wt = torch.as_tensor(weights, dtype=torch.float32)
    model.train()
    step, micro, seen, started, loss_sum, loss_n = 0, 0, 0, time.time(), 0.0, 0
    stopped = False
    while step < total and not stopped:
        for b in batches(lengths, args.batch, rng):
            ids, mask = collate(enc, b, pad, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(ids, mask)
            loss = (F.cross_entropy(logits.float(), yt[b].to(device), reduction="none") * wt[b].to(device)).sum() \
                / wt[b].sum().to(device)
            scaler.scale(loss / args.accum).backward()
            loss_sum += float(loss); loss_n += 1; micro += 1; seen += len(b)
            if micro % args.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            frac = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * frac
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0 or step == total:
                rate = (time.time() - started) / step
                log(f"{tag} step {step}/{total} loss {loss_sum / loss_n:.4f} ({seen / (time.time() - started):.1f} rows/s, "
                    f"eta {(total - step) * rate / 60:.0f} min)")
                loss_sum, loss_n = 0.0, 0
            if hours is not None and time.time() - started > hours * 3600:
                log(f"{tag} stopped early at step {step}/{total}: {hours} h limit")
                stopped = True
            if step >= total or stopped:
                break
    return dict(steps=step, total_steps=total, minutes=(time.time() - started) / 60, stopped_early=stopped)


@torch.no_grad()
def predict(model, enc, args, device, pad):
    model.eval()
    lengths = np.array([len(x) for x in enc])
    out = np.zeros((len(enc), model.head.out_features), np.float32)
    for b in batches(lengths, args.batch * 2, None, shuffle=False):
        ids, mask = collate(enc, b, pad, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out[b] = model(ids, mask).float().cpu().numpy()
    model.train()
    return out


def balanced_accuracy(y, pred, k):
    rec = [(pred[y == c] == c).mean() for c in range(k) if (y == c).any()]
    return float(np.mean(rec))


def main():
    args = parse()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    labels = json.loads(args.labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu else "cuda:0")
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id
    pool = pd.read_parquet(args.pool)
    train_df = pd.read_parquet(args.train)
    log(f"{args.model}: pool {len(pool)}, train {len(train_df)}, folds {sorted(train_df.fold.unique())}")
    enc_pool = tok(pool.text.tolist(), truncation=True, max_length=args.max_len)["input_ids"]
    enc_train = tok(train_df.text.tolist(), truncation=True, max_length=args.max_len)["input_ids"]
    log(f"tokenized: pool mean {np.mean([len(x) for x in enc_pool]):.0f} tokens, "
        f"{np.mean([len(x) == args.max_len for x in enc_pool]):.3f} truncated")
    yp = pool.label.to_numpy()
    counts = np.bincount(yp, minlength=len(labels))
    wp = (len(yp) / (len(labels) * counts))[yp] * pool.get("w", pd.Series(1.0, index=pool.index)).to_numpy()

    model = Encoder(args.model, len(labels)).to(device)
    record = dict(model=args.model, args=vars(args), pool_rows=len(pool), train_rows=len(train_df))
    record["stage1"] = train(model, enc_pool, yp, wp, args, args.lr1, args.epochs1, rng, device, pad,
                             hours=args.stage1_hours, tag="stage1")
    stage1 = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    yt = train_df.label.to_numpy()
    folds = train_df.fold.to_numpy()
    oof = np.zeros((len(train_df), len(labels)), np.float32)
    record["folds"] = {}
    for k in sorted(set(folds)) + ["full"]:
        model.load_state_dict(stage1)
        fit = np.ones(len(folds), bool) if k == "full" else folds != k
        sub = [enc_train[i] for i in np.flatnonzero(fit)]
        record["folds"][str(k)] = train(model, sub, yt[fit], np.ones(fit.sum()), args, args.lr2, args.epochs2, rng,
                                        device, pad, tag=f"stage2 fold {k}")
        if k == "full":
            break
        valid = np.flatnonzero(~fit)
        oof[valid] = predict(model, [enc_train[i] for i in valid], args, device, pad)
        acc = balanced_accuracy(yt[valid], oof[valid].argmax(1), len(labels))
        record["folds"][str(k)]["valid_balanced_accuracy"] = acc
        log(f"fold {k}: balanced accuracy {acc:.4f} on {len(valid)} rows")
    record["oof_balanced_accuracy"] = balanced_accuracy(yt, oof.argmax(1), len(labels))
    log(f"OOF balanced accuracy {record['oof_balanced_accuracy']:.4f}")
    np.savez(out / "oof.npz", ids=train_df.id.to_numpy(dtype=str), fold=folds, labels=np.asarray(labels), logits=oof)

    model.eval()
    model.encoder.save_pretrained(out / "encoder", safe_serialization=True)
    tok.save_pretrained(out / "encoder")
    save_file({"weight": model.head.weight.detach().float().cpu().contiguous(),
               "bias": model.head.bias.detach().float().cpu().contiguous()}, str(out / "head.safetensors"))
    (out / "meta.json").write_text(json.dumps(dict(labels=labels, max_len=args.max_len, pool="mean_last",
                                                   model=args.model)))
    # the check nlm-v27's infer.py runs on an 'enc' member: float32, no autocast, 40 training rows
    with torch.no_grad():
        ids, mask = collate(enc_train, list(range(40)), pad, device)
        acc40 = float((model(ids, mask).argmax(1).cpu().numpy() == yt[:40]).mean())
    record["in_sample_check_40"] = acc40
    record["minutes"] = (time.time() - T0) / 60
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))
    log(f"ENC_DONE {args.model}: OOF {record['oof_balanced_accuracy']:.4f}, in-sample 40 rows {acc40:.2f}, "
        f"{record['minutes']:.0f} min")


if __name__ == "__main__":
    main()
