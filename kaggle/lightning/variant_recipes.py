"""V28 recipe constants and the share-variant builder, shared by build_variants.py and its variants_*.py modules."""

STATUS = "FROZEN_ALLROWS_QWEN10_KIRILL10_CANDIDATE"
NOTE = (" 'note': 'New unscored combination: published Kirill10% augmented readouts after ATLAS, then own final full-text '\n"
        "         'Qwen10% on all rows before unchanged single Sinkhorn; ready readouts, no additional training.',\n")
GUARD = ("if share != 0.10 or fraction != 1.0 or recipe.get('helper_probability_share') != .10:"
         "raise ValueError('Expected all-row Qwen10% and Kirill helper10%')")
HELPER_LINE = "prior = .90*prior + .10*helper"


def pct(share):  # 0.1 -> '10', 0.125 -> '125'
    return f"{share * 100:g}".replace(".", "")


def share_variant(qwen, helper, note):
    """V28 with the Qwen and/or Kirill-helper probability share changed; everything else identical."""
    status = f"FROZEN_ALLROWS_QWEN{pct(qwen)}_KIRILL{pct(helper)}_CANDIDATE"
    helper_line = HELPER_LINE if helper == 0.10 else f"prior = {1 - helper:g}*prior + {helper:g}*helper".replace("0.", ".")
    guard = (f"if share != {qwen:g} or fraction != 1.0 or recipe.get('helper_probability_share') != {helper:g}:"
             f"raise ValueError('Expected all-row Qwen{qwen * 100:g}% and Kirill helper{helper * 100:g}%')")
    patches = [
        (f"RECIPE = {{'status': '{STATUS}',", f"RECIPE = {{'status': '{status}',"),
        (NOTE, f" 'note': {note!r},\n"),
        (f"if recipe.get('status')!='{STATUS}':", f"if recipe.get('status')!='{status}':"),
        (GUARD, guard),
    ]
    if qwen != 0.10:
        patches.append((" 'qwen_probability_share': 0.10,\n", f" 'qwen_probability_share': {qwen:g},\n"))
    if helper != 0.10:
        patches.append((" 'helper_probability_share': 0.1,\n", f" 'helper_probability_share': {helper:g},\n"))
        patches.append((HELPER_LINE, helper_line))
    spec = {"status": status, "qwen": qwen, "helper": helper,
            "must": [f"if recipe.get('status')!='{status}':", guard, helper_line, "mixed=blend_selected(prior,ids,LABELS,"
                     "selected_ids,LABELS,qwen_logits,share=share)", "probabilities=sinkhorn(mixed)"],
            "must_not": [STATUS] + ([HELPER_LINE] if helper != 0.10 else [])}
    return {"patches": patches, "spec": spec, "datasets": []}
