import re

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = (
                final_answer.split("#### ")[1].replace(",", "").replace("$", "")
            )
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            pass
        else:
            invalid_str = ["", "."]
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
