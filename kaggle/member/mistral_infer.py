"""Logits of one full-fit Mistral-7B member seed for a query CSV, without peft.

    python mistral_infer.py --backbone <dir> --member <dir> --seed 0 --query query.csv --out out.npz
                            [--deadline <time.monotonic() value>] [--start-delay 0] [--no-quant]

<member> is the output of the full-fit kernel (full_driver.py): member.json (labels, rank, alpha, max_len) and
full_seed<s>.safetensors (the trainable weights under peft's names). The base model is loaded and quantised exactly
as member_worker.py trains it (nf4, double quant, float16 compute), every float16/bfloat16 parameter is cast to
float32 as peft's prepare_model_for_kbit_training does, and each LoRA pair is applied by hand as
base(x) + B(A(x)) * alpha/rank under the same float16 autocast. So inference needs only transformers and
bitsandbytes, and does not depend on which peft version (if any) the inference image has.

Output: <out> with ids, labels (names) and float32 logits, the layout lightningv08's aligned() reads, plus <out>.json.
With --deadline the worker exits with code 3 and writes nothing if it cannot finish by then: before loading when
fewer than 10 minutes are left, and while predicting when the token-proportional projection overruns (rows run
longest first, so the projection starts pessimistic).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PREFIX = "base_model.model."
SCORE = "score.modules_to_save.default.weight"
OUT_OF_TIME = 3


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--member", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--deadline", type=float, default=None)
    p.add_argument("--start-delay", type=float, default=0)
    p.add_argument("--eval-tokens", type=int, default=4096)
    p.add_argument("--no-quant", action="store_true")
    return p.parse_args()


class LoRALinear(torch.nn.Module):
    """A frozen (quantised) linear layer plus a LoRA update, as peft computes it at inference (dropout off)."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.scale = scale

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b) * self.scale


def out_of_time(deadline, message):
    print(f"MISTRAL_OUT_OF_TIME: {message}", flush=True)
    sys.exit(OUT_OF_TIME)


def load(args, member, tokenizer, device):
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    kwargs = dict(num_labels=len(member["labels"]), pad_token_id=tokenizer.pad_token_id)
    if args.no_quant:
        model = AutoModelForSequenceClassification.from_pretrained(args.backbone, dtype=torch.float32, **kwargs)
        model.to(device)
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.backbone, quantization_config=quant, dtype=torch.float16, device_map={"": 0}, **kwargs)
        for param in model.parameters():  # prepare_model_for_kbit_training: non-quantised weights to float32
            if param.dtype in (torch.float16, torch.bfloat16):
                param.data = param.data.to(torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    weights = load_file(str(Path(args.member) / f"full_seed{args.seed}.safetensors"))
    if not all(name.startswith(PREFIX) for name in weights):
        raise SystemExit("unexpected weight names")
    weights = {name[len(PREFIX):]: tensor for name, tensor in weights.items()}
    score = weights.pop(SCORE)
    if not isinstance(model.score, torch.nn.Linear) or model.score.weight.shape != score.shape or model.score.bias is not None:
        raise SystemExit(f"score head differs: {model.score} vs {tuple(score.shape)}")
    model.score.weight.data = score.to(device=model.score.weight.device, dtype=model.score.weight.dtype)
    scale = member["alpha"] / member["rank"]
    modules = dict(model.named_modules())
    pairs = sorted({name.rsplit(".lora_", 1)[0] for name in weights})
    for path in pairs:
        a = weights.pop(f"{path}.lora_A.default.weight")
        b = weights.pop(f"{path}.lora_B.default.weight")
        parent, child = path.rsplit(".", 1)
        base = modules[path]
        if a.shape[0] != member["rank"] or b.shape[1] != member["rank"] or a.shape[1] != base.in_features \
                or b.shape[0] != base.out_features:
            raise SystemExit(f"{path}: LoRA shapes {tuple(a.shape)} {tuple(b.shape)} vs {base}")
        dev = next(base.parameters()).device
        setattr(modules[parent], child, LoRALinear(base, a.to(dev), b.to(dev), scale))
    if weights or len(pairs) != member["lora_modules"]:
        raise SystemExit(f"{len(pairs)} LoRA modules applied (expected {member['lora_modules']}); unused {sorted(weights)[:5]}")
    model.eval()
    return model, len(pairs)


def main():
    args = parse()
    started = time.monotonic()
    member = json.loads((Path(args.member) / "member.json").read_text())
    query = pd.read_csv(args.query, dtype={"id": str}, keep_default_na=False)
    if args.start_delay:
        time.sleep(args.start_delay)  # stagger two workers' model loads (host RAM)
    if args.deadline is not None and time.monotonic() + 600 > args.deadline:
        out_of_time(args.deadline, f"{(args.deadline - time.monotonic()) / 60:.1f} min left before loading")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if tokenizer.pad_token_id is None:  # as member_worker.py
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cpu" if args.no_quant else "cuda:0")
    model, applied = load(args, member, tokenizer, device)
    loaded = time.monotonic()

    encoded = tokenizer(query.text.fillna("").astype(str).tolist(), add_special_tokens=True, truncation=True,
                        max_length=member["max_len"])["input_ids"]
    lengths = np.array([len(x) for x in encoded])
    order = [i for i in np.argsort(-lengths, kind="stable")]
    plan, current = [], []
    for i in order:  # member_worker.eval_batches: longest first, at most --eval-tokens padded tokens per batch
        if current and (len(current) + 1) * lengths[current[0]] > args.eval_tokens:
            plan.append(current)
            current = []
        current.append(i)
    if current:
        plan.append(current)
    total_tokens, done_tokens = int(lengths.sum()), 0
    logits = np.zeros((len(query), len(member["labels"])), np.float32)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=not args.no_quant)
    t0 = time.monotonic()
    with torch.no_grad(), autocast:
        for step, idx in enumerate(plan):
            width = int(lengths[idx[0]])
            ids = torch.full((len(idx), width), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                ids[row, :lengths[i]] = torch.tensor(encoded[i])
                mask[row, :lengths[i]] = 1
            logits[idx] = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float().cpu().numpy()
            done_tokens += int(lengths[idx].sum())
            if args.deadline is not None:
                now = time.monotonic()
                projected = now + (now - t0) / done_tokens * (total_tokens - done_tokens)
                if now > args.deadline or (step >= 4 and projected > args.deadline):
                    out_of_time(args.deadline, f"batch {step + 1}/{len(plan)}, projected end "
                                               f"{(projected - args.deadline) / 60:+.1f} min past the deadline")
    if not np.isfinite(logits).all():
        raise SystemExit("non-finite logits")
    np.savez(args.out, ids=query.id.to_numpy(dtype=str), labels=np.asarray(member["labels"]), logits=logits)
    record = dict(rows=len(query), seed=args.seed, lora_modules=applied, batches=len(plan), tokens=total_tokens,
                  load_seconds=loaded - started, predict_seconds=time.monotonic() - t0,
                  peak_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0)
    Path(args.out + ".json").write_text(json.dumps(record, indent=1))
    print("MISTRAL_DONE " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
