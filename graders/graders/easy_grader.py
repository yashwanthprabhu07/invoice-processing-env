def grade_easy(state):
    correct = 0
    total = 3
    for k, v in state["true_fields"].items():
        if state["extracted_fields"].get(k) == v:
            correct += 1
    raw = correct / total
    return max(0.01, min(0.99, raw))