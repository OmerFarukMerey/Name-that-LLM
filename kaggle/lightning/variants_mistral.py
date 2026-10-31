"""ntl-v28-mistral-5 / -10: V28 plus our full-fit Mistral-7B member at 5% / 10% (training track, Option A).

The member is the phase-1 pilot recipe (LoRA r16 sequence classifier on Mistral-7B-v0.1, 2 epochs, max 512 tokens)
trained on all 2,699 official rows, two seeds (omerfarukmerey/ntl-member-mistral7b-full, kaggle/member/full_driver.py).
Out of fold it scored only 0.422 alone, but its errors differ from the ensemble's (top-guess agreement 0.41 with a
bank head, against 0.70 between two Qwen heads) and it added +0.028 [+0.014, +0.042] to the OOF bank's blend.

In the run script, after the full-text Qwen worker and before the unchanged Qwen blend, mistral_member() (written into
RUNTIME with mistral_infer.py) scores every test row with both seeds, one per GPU, and averages their probabilities M.
Then prior = (1-w)*prior + w*M, the way the Kirill helper enters, so the nominal shares become
(1-w) x V28's prior, w x Mistral, then 10% Qwen on top. If the member fails, or cannot finish by 8.5 h after the
notebook started, mistral_member() returns None and the prior stays V28's: the submission is V28's own.
Qwen 10% and helper 10% stay; the status string names the variant so the guards identify it.
The self-test (e) (selftest_mistral_check.py) pins the weights by sha256, runs the member for real in this image on
24 training rows plus the public test rows against the training kernel's own logits, and checks the fallback.
"""
from variant_recipes import STATUS, share_variant

RUNTIME = "/kaggle/temp/ntl636-fulltext-qwen-external3200-runtime"
MEMBER = "omerfarukmerey/ntl-member-mistral7b-full"
MISTRAL_MODEL = "mistral-ai/mistral/pyTorch/7b-v0.1-hf/1"
IMPORT = "from mistral_member import mistral_member"
CALL = "mistral,mistral_report=mistral_member(ids,LABELS,query_path,work,logs,target)"
REPORT = "print(json.dumps(mistral_report),flush=True)"
BLEND = "    mixed=blend_selected(prior,ids,LABELS,selected_ids,LABELS,qwen_logits,share=share)\n"


def mistral_variant(share):
    pct = f"{share * 100:g}"
    status = STATUS.replace("_CANDIDATE", f"_MISTRAL{pct}_CANDIDATE")  # FROZEN_ALLROWS_QWEN10_KIRILL10_MISTRAL5_CANDIDATE
    note = (f"lightningv08 V28 plus our full-fit Mistral-7B LoRA member at {pct}% of the prior (before the Qwen blend); "
            "Qwen 10% and Kirill helper 10% unchanged.")
    base = share_variant(0.10, 0.10, note)
    if sum(STATUS in new for _, new in base["patches"]) != 2:
        raise SystemExit("share_variant no longer patches the status in exactly two places")
    patches = [(old, new.replace(STATUS, status)) for old, new in base["patches"]]
    mix = f"if mistral is not None:prior={1 - share:g}*prior+{share:g}*mistral".replace("0.", ".")
    patches += [
        ("from query_allocation import select_low_margin,blend_selected\n",
         "from query_allocation import select_low_margin,blend_selected\n" + IMPORT + "\n"),
        ("    qwen_logits=aligned(work/'qwen.npz',selected_ids)\n" + BLEND,
         "    qwen_logits=aligned(work/'qwen.npz',selected_ids)\n"
         f"    # ntl-v28-mistral-{pct}: full-fit Mistral-7B member mixed into the prior like the helper; None keeps V28's\n"
         f"    {CALL}\n"
         f"    {REPORT}\n"
         f"    {mix}\n" + BLEND),
        ("qwen_probability_share=share,qwen_query_fraction=fraction,",
         "qwen_probability_share=share,qwen_query_fraction=fraction,mistral_used=mistral is not None,"),
    ]
    spec = dict(base["spec"], status=status, mistral=share,
                must=[line.replace(STATUS, status) for line in base["spec"]["must"]]
                + [IMPORT, CALL, REPORT, mix, "mistral_used=mistral is not None,"],
                must_not=[STATUS])
    return {"patches": patches, "spec": spec, "datasets": [], "kernels": [MEMBER], "models": [MISTRAL_MODEL],
            "cells": [(f"%%writefile {RUNTIME}/mistral_member.py\n", "mistral_member.py"),
                      (f"%%writefile {RUNTIME}/mistral_infer.py\n", "../member/mistral_infer.py")],
            "extra_check": "selftest_mistral_check.py"}


VARIANTS = {"ntl-v28-mistral-5": mistral_variant(0.05), "ntl-v28-mistral-10": mistral_variant(0.10)}
