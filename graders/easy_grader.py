def grade_easy(state):
    try:
        correct = 0
        total = max(1, len(state["true_fields"]))
        for k, v in state["true_fields"].items():
            if state["extracted_fields"].get(k) == v:
                correct += 1
        raw = correct / total
    except Exception:
        raw = 0.5
    return float(max(0.01, min(0.99, float(raw))))