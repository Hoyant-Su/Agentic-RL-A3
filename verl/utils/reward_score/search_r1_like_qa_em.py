import random
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()

def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")
    return opening_tags, closing_tags

def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:
                score = score / 4
                return score
            return score
        else:
            return format_score

def compute_score_subem(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
