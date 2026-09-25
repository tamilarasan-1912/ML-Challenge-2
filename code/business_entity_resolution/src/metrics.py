def f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)

def macro_f05(pred, truth):
    scores = []
    for sid, true_ids in truth.items():
        predicted = pred.get(sid, set())
        if not true_ids:
            scores.append(1.0 if not predicted else 0.0)
            continue
        tp = len(predicted & true_ids)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(true_ids)
        scores.append(f05(precision, recall))
    return sum(scores) / len(scores) if scores else 0.0
