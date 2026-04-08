def _clamp_score(raw):
    return float(max(0.1, min(0.9, float(raw))))


def grade_easy(state):
    correct = 0
    total = 3
    for k, v in state["true_fields"].items():
        if state["extracted_fields"].get(k) == v:
            correct += 1
    raw = correct / total
    return _clamp_score(raw)