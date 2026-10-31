"""One GPU's share of the pilot: LoRA sequence classifier, one model per outer fold, out-of-fold logits saved.

    python member_worker.py --data member_data.parquet --backbone <dir> --folds 0,2,4 --out <dir> [--no-quant]

The driver (member_driver.py) starts one worker per GPU with CUDA_VISIBLE_DEVICES set. Each fold trains on the rows
of the other folds and writes <out>/fold<k>.npz (valid ids, float32 logits) and <out>/fold<k>.json (timings).
Fold -1 is the full fit (full_driver.py): it trains on every row, predicts a fixed 256-row in-sample check set and
also writes <out>/full_seed<s>.safetensors, the trainable weights only (LoRA A/B and the score head, float32, peft's
parameter names), which mistral_infer.py applies to the quantised base without peft.
--no-quant runs in float32 on CPU without bitsandbytes: the local smoke test with a tiny random model.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TARGETS = {  # LoRA on every linear layer of the decoder blocks; the score head is trained in full
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--eval-tokens", type=int, default=4096)  # padded tokens per prediction batch
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-quant", action="store_true")
    return p.parse_args()


def load_model(args, n_labels, tokenizer):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(args.backbone)
    arch = config.model_type
    if arch not in TARGETS:
        raise SystemExit(f"no LoRA targets for model_type {arch!r}")
    kwargs = dict(num_labels=n_labels, pad_token_id=tokenizer.pad_token_id)
    if args.no_quant:
        model = AutoModelForSequenceClassification.from_pretrained(args.backbone, dtype=torch.float32, **kwargs)
    else:
        from peft import prepare_model_for_kbit_training
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.backbone, quantization_config=quant, dtype=torch.float16, device_map={"": 0}, **kwargs)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.pad_token_id = tokenizer.pad_token_id
    lora = LoraConfig(task_type="SEQ_CLS", r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=TARGETS[arch], modules_to_save=["score"])
    model = get_peft_model(model, lora)
    for name, param in model.named_parameters():  # trainable weights in float32 for the fp16 grad scaler
        if param.requires_grad:
            param.data = param.data.float()
    return model


def batches(lengths, size, rng=None):
    """Index batches of similar length. With rng: shuffled, sorted within chunks of 16 batches, chunks shuffled."""
    order = np.arange(len(lengths)) if rng is None else rng.permutation(len(lengths))
    chunk = len(order) if rng is None else size * 16
    out = []
    for start in range(0, len(order), chunk):
        part = order[start:start + chunk]
        part = part[np.argsort(-lengths[part], kind="stable")]
        out += [part[i:i + size] for i in range(0, len(part), size)]
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def collate(encoded, idx, pad_id, device):
    width = max(len(encoded[i]) for i in idx)
    ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for row, i in enumerate(idx):
        ids[row, :len(encoded[i])] = torch.tensor(encoded[i])
        mask[row, :len(encoded[i])] = 1
    return ids.to(device), mask.to(device)


def eval_batches(lengths, budget):
    """Longest rows first; each batch holds at most `budget` padded tokens."""
    out, current = [], []
    for i in np.argsort(-lengths, kind="stable"):
        if current and (len(current) + 1) * lengths[current[0]] > budget:
            out.append(np.array(current))
            current = []
        current.append(i)
    return out + [np.array(current)] if current else out


def predict(model, encoded, lengths, args, pad_id, device, autocast, n_labels):
    """Logits for every row; a batch that runs out of GPU memory is split in two and retried."""
    logits = np.zeros((len(encoded), n_labels), np.float32)
    pending = eval_batches(lengths, args.eval_tokens)
    with torch.no_grad(), autocast:
        while pending:
            idx = pending.pop(0)
            ids, mask = collate(encoded, idx, pad_id, device)
            try:
                logits[idx] = model(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if len(idx) == 1:
                    raise
                del ids, mask
                torch.cuda.empty_cache()
                print(f"prediction batch of {len(idx)} rows out of memory; splitting", flush=True)
                pending[:0] = [idx[:len(idx) // 2], idx[len(idx) // 2:]]
    return logits


def preflight(model, optimizer, fit_ids, fit_len, valid_ids, valid_len, args, pad_id, device, autocast, n_labels):
    """One training step on the longest training batch (learning rate 0) and one prediction on the longest
    prediction batch, so a memory error shows up in minutes rather than after a whole fold of training."""
    worst = np.argsort(-fit_len, kind="stable")[:args.batch]
    ids, mask = collate(fit_ids, worst, pad_id, device)
    with autocast:
        loss = model(input_ids=ids, attention_mask=mask).logits.float().logsumexp(-1).mean()
    loss.backward()
    rates = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    optimizer.step()  # allocates the Adam state, changes no weight
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    worst = eval_batches(valid_len, args.eval_tokens)[0]
    predict(model, [valid_ids[i] for i in worst], valid_len[worst], args, pad_id, device, autocast, n_labels)
    model.train()
    optimizer.state.clear()
    del ids, mask, loss
    if device.type == "cuda":
        print(f"preflight ok: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB of "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}", flush=True)
        torch.cuda.empty_cache()


def run_fold(args, data, fold, tokenizer, device, out):
    started = time.time()
    full = fold < 0
    torch.manual_seed(args.seed + max(fold, 0))
    rng = np.random.default_rng(args.seed + max(fold, 0))
    if full:  # the check rows are training rows: they test the saved weights, not generalisation
        fit, valid = data, data.sample(n=min(256, len(data)), random_state=0)
    else:
        fit, valid = data[data.fold != fold], data[data.fold == fold]
    tag = f"full_seed{args.seed}" if full else f"fold{fold}"
    encode = lambda texts: tokenizer(list(texts), add_special_tokens=True, truncation=True,
                                     max_length=args.max_len)["input_ids"]
    fit_ids, valid_ids = encode(fit.text), encode(valid.text)
    fit_len, valid_len = np.array([len(x) for x in fit_ids]), np.array([len(x) for x in valid_ids])
    y = torch.tensor(fit.label.to_numpy())

    model = load_model(args, int(data.label.max()) + 1, tokenizer)
    if args.no_quant:
        model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(fit) / (args.batch * args.accum))
    total = max(1, int(round(steps_per_epoch * args.epochs)))
    warmup = max(1, int(0.06 * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_quant)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=not args.no_quant)

    model.train()
    preflight(model, optimizer, fit_ids, fit_len, valid_ids, valid_len, args, tokenizer.pad_token_id, device, autocast,
              int(data.label.max()) + 1)
    step, seen_tokens, losses, bad = 0, 0, [], 0
    tick = time.time()
    while step < total:
        plan = batches(fit_len, args.batch, rng)
        for b in range(0, len(plan) - args.accum + 1, args.accum):
            for idx in plan[b:b + args.accum]:
                ids, mask = collate(fit_ids, idx, tokenizer.pad_token_id, device)
                with autocast:
                    logits = model(input_ids=ids, attention_mask=mask).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, y[idx].to(device)) / args.accum
                if not torch.isfinite(loss):
                    bad += 1
                    if bad > 10:
                        raise SystemExit(f"fold {fold}: loss not finite at step {step}")
                    continue
                scaler.scale(loss).backward()
                losses.append(loss.item() * args.accum)
                seen_tokens += int(mask.sum())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            step += 1
            if step % 20 == 0 or step == total:
                rate = seen_tokens / (time.time() - tick)
                print(f"fold {fold} step {step}/{total} loss {np.mean(losses[-40:]):.4f} {rate:.0f} tok/s "
                      f"elapsed {(time.time() - started) / 60:.1f} min", flush=True)
            if step >= total:
                break
    train_minutes = (time.time() - started) / 60

    optimizer.state.clear()  # free the Adam state before predicting
    del optimizer, schedule, scaler, params
    if not args.no_quant:
        torch.cuda.empty_cache()
    model.eval()
    logits = predict(model, valid_ids, valid_len, args, tokenizer.pad_token_id, device, autocast,
                     int(data.label.max()) + 1)
    if not np.isfinite(logits).all():
        raise SystemExit(f"fold {fold}: non-finite validation logits")
    accuracy = float((logits.argmax(1) == valid.label.to_numpy()).mean())
    np.savez(out / f"{tag}.npz", ids=valid.id.to_numpy(dtype=str), logits=logits)
    if full:
        from safetensors.torch import save_file
        weights = {name: param.detach().float().cpu().contiguous()
                   for name, param in model.named_parameters() if param.requires_grad}
        save_file(weights, str(out / f"{tag}.safetensors"),
                  metadata=dict(rank=str(args.rank), alpha=str(2 * args.rank), max_len=str(args.max_len)))
    record = dict(fold=fold, seed=args.seed, train_rows=len(fit), valid_rows=len(valid), steps=total,
                  train_minutes=train_minutes, total_minutes=(time.time() - started) / 60,
                  final_loss=float(np.mean(losses[-40:])), valid_accuracy=accuracy,
                  truncated=int((fit_len >= args.max_len).sum() + (valid_len >= args.max_len).sum()))
    (out / f"{tag}.json").write_text(json.dumps(record, indent=1))
    print(f"fold {fold} done: {json.dumps(record)}", flush=True)
    del model
    if not args.no_quant:
        torch.cuda.empty_cache()


def main():
    args = parse()
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cpu" if args.no_quant else "cuda:0")
    for fold in [int(f) for f in args.folds.split(",")]:
        run_fold(args, data, fold, tokenizer, device, out)


if __name__ == "__main__":
    main()
