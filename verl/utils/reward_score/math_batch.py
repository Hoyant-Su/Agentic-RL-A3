from .math import compute_score

def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos):
    return [
        compute_score(solution_str, ground_truth)
        for solution_str, ground_truth in zip(solution_strs, ground_truths)
    ]
