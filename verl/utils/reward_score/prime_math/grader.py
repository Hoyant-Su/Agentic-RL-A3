
import contextlib
import math
import re
from math import isclose
from typing import Union
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from verl.utils.py_functional import timeout_limit

def is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num
        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None

def normalize(answer, pi) -> str:
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer))
        or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")
    answer = handle_base(answer)
    answer = handle_pi(answer, pi)
    return answer

def handle_base(x) -> str:
    if isinstance(x, str) and "_" in x:
        x = x.split("_")[0]
        x = float(x)
        return int(x)
    return x

def handle_pi(string, pi):
    if isinstance(string, str) and "\pi" in string:
        idx = string.find("\pi")
        while idx != -1:
            if idx > 0 and string[idx - 1].isdigit():
                string = string[:idx] + f"*{pi}" + string[idx + 3 :]
            else:
                string = string[:idx] + f"1*{pi}" + string[idx + 3 :]
            idx = string.find("\pi", idx + 1)
        with contextlib.suppress(Exception):
            string = eval(string)
    return string

def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    pi: float = math.pi,
) -> bool:
    prediction = normalize(prediction, pi)
    reference = normalize(reference, pi)
    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True
    try:
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]
            gt_result = (
                [reference / 100, reference, reference * 100]
                if include_percentage
                else [reference]
            )
            for item in gt_result:
                try:
                    if isclose(item, prediction, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass
    if not prediction and prediction not in [0, False]:
        return False
    reference = str(reference).strip()
    prediction = str(prediction).strip()
    prediction = format_intervals(prediction)
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ]
        ):
            return True
    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]
        if len(pred_parts) == len(ref_parts):
            return bool(
                all(
                    [
                        math_equal(
                            pred_parts[i], ref_parts[i], include_percentage, tolerance
                        )
                        for i in range(len(pred_parts))
                    ]
                )
            )
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ]
        ):
            return True
    if "\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items) and all(
                [
                    math_equal(pred, ref, include_percentage, tolerance)
                    for ref, pred in zip(ref_matrix_items, pred_matrix)
                ]
            ):
                return True
        except Exception:
            pass
    elif (
        "\begin{pmatrix}" in reference
        and prediction.startswith("[")
        and prediction.endswith("]")
    ):
        if isinstance(eval(prediction), list):
            try:
                pred_matrix = eval(prediction)
                ref_matrix_items = (
                    reference.lstrip("\\begin{pmatrix}")
                    .lstrip("\begin{pmatrix}")
                    .rstrip("\\end{pmatrix}")
                    .rstrip("\end{pmatrix}")
                )
                ref_matrix_items = ref_matrix_items.split("\\")
                ref_matrix_items = [
                    row.split("&") if "&" in row else row for row in ref_matrix_items
                ]
                if len(pred_matrix) == len(ref_matrix_items) and all(
                    [
                        math_equal(pred, ref, include_percentage, tolerance)
                        for ref, pred in zip(ref_matrix_items, pred_matrix)
                    ]
                ):
                    return True
            except Exception:
                pass
    return symbolic_equal(prediction, reference, tolerance, timeout)

def symbolic_equal(a, b, tolerance, timeout=10.0):
    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with timeout_limit(seconds=timeout):
                    return f(s)
            except TimeoutError:
                print(f"Parsing timed out for {s}")
                continue
            except Exception:
                continue
        return s

    a = _parse(a)
    b = _parse(b)
    try:
        with timeout_limit(seconds=timeout):
            if simplify(a - b) == 0:
                return True
    except TimeoutError:
        print(f"Simplification timed out for {a} - {b}")
        pass
    except Exception:
        pass
    try:
        with timeout_limit(seconds=timeout):
            if isclose(N(a), N(b), rel_tol=tolerance):
                return True
    except TimeoutError:
        print(f"Numerical evaluation timed out for {a}, {b}")
        pass
    except Exception:
        pass
    return False

def format_intervals(prediction):
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }
    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)
            if key == "Interval(":
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":
                return f"({inner_content}]"
            elif key == "Interval.open(":
                return f"({inner_content})"
    return prediction
