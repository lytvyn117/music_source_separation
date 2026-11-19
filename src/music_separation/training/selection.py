"""Rank both weight sets and retain the original EMA early-stopping rule."""
import math


def decision(history, step, metrics, policy):
    prior = history[-1]["selection"] if history else {}
    overall = prior.get("best_overall")
    qualified = prior.get("best_qualified")
    candidates = {}
    # Exact ties retain the earlier checkpoint, with EMA first within the same update.
    for weights in ("ema", "raw"):
        values = metrics[weights]
        failures = []
        for name, limits in policy["guardrails"].items():
            value = values[name]
            if not math.isfinite(value) or any(
                (value < limit if direction == "min" else value > limit)
                for direction, limit in limits.items()
            ):
                failures.append(name)
        score = values[policy["primary"]]
        if not math.isfinite(score):
            failures.append(policy["primary"])
        eligible = not failures
        candidates[weights] = {"eligible": eligible, "failed_guardrails": failures}
        candidate = {"step": step, "weights": weights, "vocal_sdr_db": score}
        if math.isfinite(score) and (overall is None or score > overall["vocal_sdr_db"]):
            overall = candidate
        if eligible and (qualified is None or score > qualified["vocal_sdr_db"]):
            qualified = candidate
    anchor = prior.get("anchor_step")
    anchor_score = prior.get("anchor_score")
    stale = prior.get("stale_validations", 0)
    score = metrics["ema"][policy["primary"]]
    if candidates["ema"]["eligible"] and (anchor is None or score - anchor_score >= policy["min_delta_db"]):
        anchor, anchor_score = step, score
        stale = 0
    elif anchor is not None:
        stale += 1
    return {
        "candidates": candidates, "best_overall": overall, "best_qualified": qualified,
        "anchor_step": anchor, "anchor_score": anchor_score,
        "stale_validations": stale,
        "stop": step >= policy["min_updates"] and stale >= policy["patience_validations"],
    }
