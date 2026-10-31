"""ntl-v28-dual: V28 plus Roman Tamrazov's dual head for the full-text Qwen3.8-27B (romantamrazov/name-that-llm-who-am-i V22).

The 27B checkpoint ships a trained prompt-centred head, model.centered (Linear 40960->18), that V28 loads but never calls.
Inside topic clusters of 8-18 rows (topic_groups at .90) the worker centres the pooled 40960-d features on the cluster
mean and scores them with that head; the run script then feeds dual=(1-a)*softmax(plain)+a*softmax(centered) (a=.35
where the heads agree on argmax, .18 otherwise) into the unchanged 10% Qwen blend. Every other row gets V28's plain
logits bit for bit. Qwen 10% and helper 10% stay; only the status string changes, so the guards identify the variant.

Code lives in dual_head.py (selftest_dual_runtime.py, written into RUNTIME). The worker imports it at the top and runs
its smoke test and the real-text clustering before the 27B model loads, so an import or logic error costs seconds.
The self-test (e) (selftest_dual_check.py) runs selftest_dual_child.py in a child process: the patched worker's main()
end to end with a mocked classifier, then the patched run-script block, both against an independent reference.
"""
from variant_recipes import STATUS, share_variant

DUAL_STATUS = STATUS.replace("QWEN10_", "QWEN10DUAL_")  # FROZEN_ALLROWS_QWEN10DUAL_KIRILL10_CANDIDATE
RUNTIME = "/kaggle/temp/ntl636-fulltext-qwen-external3200-runtime"
CHILD = "/kaggle/temp/ntl_dual_selftest_child.py"  # outside RUNTIME: (d) counts RUNTIME modules
NOTE = ("lightningv08 V28 plus the Qwen3.8-27B checkpoint's own prompt-centred head (dual head after Roman Tamrazov) on "
        "rows in 8-18-row topic clusters; Qwen 10% and Kirill helper 10% unchanged.")

OLD_BLEND = "    mixed=blend_selected(prior,ids,LABELS,selected_ids,LABELS,qwen_logits,share=share)\n"
V28_MIX = "v28_mix=blend_selected(prior,ids,LABELS,selected_ids,LABELS,qwen_logits,share=share)"
NEW_BLEND = "mixed=blend_selected(prior,ids,LABELS,selected_ids,LABELS,qwen_input,share=share)"
IMPORT = "from dual_head import dual_head_logits"
CENTERED = "qwen_centered=aligned(work/'qwen.npz',selected_ids,field='centered_logits')"
DUAL = "qwen_input,dual_report=dual_head_logits(qwen_logits,qwen_centered,centered_valid)"
CHANGED = "dual_report['qwen_dual_changed_rows']=int((sinkhorn(v28_mix).argmax(1)!=probabilities.argmax(1)).sum())"

RUN_PATCHES = [
    ("from query_allocation import select_low_margin,blend_selected\n",
     "from query_allocation import select_low_margin,blend_selected\n" + IMPORT + "\n"),
    ("    qwen_logits=aligned(work/'qwen.npz',selected_ids)\n" + OLD_BLEND,
     "    qwen_logits=aligned(work/'qwen.npz',selected_ids)\n"
     "    # ntl-v28-dual: the 27B checkpoint's prompt-centred head on 8-18-row topic clusters (dual_head.py)\n"
     f"    {CENTERED}\n"
     "    with np.load(work/'qwen.npz',allow_pickle=False) as z:\n"
     "        if z['ids'].astype(str).tolist()!=selected_ids or z['centered_valid'].shape!=(len(selected_ids),):"
     "raise ValueError('Qwen dual-head rows differ')\n"
     "        centered_valid=z['centered_valid']\n"
     f"    {DUAL}\n"
     f"    {V28_MIX}\n"
     f"    {NEW_BLEND}\n"
     "    print(json.dumps(dict(event='QWEN_DUAL_HEAD_BLEND',**dual_report)),flush=True)\n"),
    ("    probabilities=sinkhorn(mixed)\n", f"    probabilities=sinkhorn(mixed)\n    {CHANGED}\n"),
    ("qwen_probability_share=share,qwen_query_fraction=fraction,",
     "qwen_probability_share=share,qwen_query_fraction=fraction,**dual_report,"),
]
WORKER_PATCHES = [
    ("from qwen_fulltext_inference import FullTextClassifier\n",
     "from qwen_fulltext_inference import FullTextClassifier\n"
     "from dual_head import centered_head_logits,dual_groups,smoke_test,summary\n"),
    ("    started=time.monotonic()\n    model=FullTextClassifier(",
     "    started=time.monotonic()\n"
     "    smoke=smoke_test()  # ntl-v28-dual: helper and clustering fail here, before the 27B model loads\n"
     "    groups,valid=dual_groups(texts)\n"
     "    dual=summary(groups,valid)\n"
     "    print(json.dumps(dict(event='QWEN_DUAL_HEAD_GROUPS',smoke=smoke,**dual)),flush=True)\n"
     "    model=FullTextClassifier("),
    ("    logits=model.predict(texts,batch_size=3)\n"
     "    assert logits.shape==(len(frame),18) and np.isfinite(logits).all()\n"
     "    np.savez_compressed(a.out,ids=frame.id.to_numpy(dtype=str),labels=np.asarray(model.labels),logits=logits)\n"
     "    Path(a.out+'.json').write_text(json.dumps(dict(rows=len(frame),",
     "    logits,features=model.predict(texts,batch_size=3,return_features=True)\n"
     "    assert logits.shape==(len(frame),18) and np.isfinite(logits).all()\n"
     "    def head(layer):\n"
     "        def score(block):\n"
     "            with torch.inference_mode():return layer(torch.from_numpy(block).to(model.device)).float().cpu().numpy()\n"
     "        return score\n"
     "    centered_logits,recheck=centered_head_logits(logits,features,groups,valid,head(model.centered),head(model.plain))\n"
     "    del features\n"
     "    dual.update(centered_head_agree_rows=int((centered_logits.argmax(1)==logits.argmax(1))[valid].sum()),"
     "plain_recheck_max_abs=recheck)\n"
     "    np.savez_compressed(a.out,ids=frame.id.to_numpy(dtype=str),labels=np.asarray(model.labels),logits=logits,"
     "centered_logits=centered_logits,centered_valid=valid.astype(np.uint8),centered_groups=groups)\n"
     "    print(json.dumps(dict(event='QWEN_DUAL_HEAD_DONE',**dual)),flush=True)\n"
     "    Path(a.out+'.json').write_text(json.dumps(dict(dual_head=dual,rows=len(frame),"),
]


def dual_variant():
    base = share_variant(0.10, 0.10, NOTE)
    patches = [(old, new.replace(STATUS, DUAL_STATUS)) for old, new in base["patches"]]
    if sum(STATUS in new for _, new in base["patches"]) != 2:
        raise SystemExit("share_variant no longer patches the status in exactly two places")
    must = [line.replace(STATUS, DUAL_STATUS) for line in base["spec"]["must"] if "qwen_logits,share=share)" not in line]
    spec = dict(base["spec"], status=DUAL_STATUS,
                must=must + [IMPORT, CENTERED, DUAL, V28_MIX, NEW_BLEND, CHANGED, "qwen_query_fraction=fraction,**dual_report,"],
                must_not=[STATUS, OLD_BLEND.strip()],
                worker_must=["from dual_head import centered_head_logits,dual_groups,smoke_test,summary",
                             "smoke=smoke_test()", "groups,valid=dual_groups(texts)",
                             "logits,features=model.predict(texts,batch_size=3,return_features=True)",
                             "head(model.centered),head(model.plain)", "centered_valid=valid.astype(np.uint8)"],
                child=CHILD)
    return {"patches": patches + RUN_PATCHES + WORKER_PATCHES, "spec": spec, "datasets": [],
            "cells": [(f"%%writefile {RUNTIME}/dual_head.py\n", "selftest_dual_runtime.py"),
                      (f"%%writefile {CHILD}\n", "selftest_dual_child.py")],
            "extra_check": "selftest_dual_check.py"}


VARIANTS = {"ntl-v28-dual": dual_variant()}
