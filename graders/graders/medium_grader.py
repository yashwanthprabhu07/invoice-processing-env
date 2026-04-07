def grade_medium(state):
    correct = 0
    total = 3
    for k, v in state["true_fields"].items():
        if state["extracted_fields"].get(k) == v:
            correct += 1
    accuracy = correct / total
    penalty = 0.1 * len(state["validation_errors"])
    raw = max(0.0, accuracy - penalty)
    return max(0.01, min(0.99, raw))