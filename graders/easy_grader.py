def grade_easy(state):
    correct = 0
    total = 3
    for k, v in state["true_fields"].items():
        if state["extracted_fields"].get(k) == v:
            correct += 1
    return correct / total