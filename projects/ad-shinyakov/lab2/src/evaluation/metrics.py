def score_ranking(ranking, judgment, ks=(5, 10)):
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate chunk IDs")
    relevant = set(judgment["relevant_chunk_ids"])
    groups = [
        {match["chunk_id"] for match in group["matches"]}
        for group in judgment["evidence_groups"]
    ]
    if not relevant or not groups:
        raise ValueError("Question has no positive judgments")
    result = {}
    for k in ks:
        if k <= 0:
            raise ValueError("k must be positive")
        top = ranking[:k]
        hits = len(relevant.intersection(top))
        covered = sum(bool(group.intersection(top)) for group in groups)
        first = next((i for i, cid in enumerate(top, 1) if cid in relevant), None)
        result.update({
            f"recall@{k}": hits / len(relevant),
            f"precision@{k}": hits / k,
            f"mrr@{k}": 1 / first if first else 0.0,
            f"evidence_coverage@{k}": covered / len(groups),
            f"complete_evidence@{k}": float(covered == len(groups)),
        })
    return result


def aggregate(rows):
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
