def grade_hard(state):
    correct = 0
    total = 3
    for k, v in state["true_fields"].items():
        if state["extracted_fields"].get(k) == v:
            correct += 1
    accuracy = correct / total
    penalty = 0.1 * len(state["validation_errors"])
    fraud_bonus = 0.2 if state["fraud_detected"] else -0.2
    return max(0.0, accuracy - penalty) + fraud_bonus