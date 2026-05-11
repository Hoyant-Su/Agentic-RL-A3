import argparse
import re
import requests
from sglang import assistant, function, gen, user
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

DEFAULT_API_BASE_URL = "http://127.0.0.1:30003"
DEFAULT_MODEL = "default"
DEFAULT_TIMEOUT = 60
DEFAULT_BATCH_SIZE = 32
DEMO_PAIRS = [
    ("12.", "twelve."),
    ("11.", "twelve."),
    ("yes", "true"),
    ("no", "false"),
    ("consistent", "inconsistent"),
    ("3.14", "3.140"),
    ("The capital of France is Paris.", "Paris is the capital of France."),
    ("The answer is 42.", "The answer is forty-two."),
    ("The patient has diabetes.", "The patient does not have diabetes."),
]

def build_prompt(text_a: str, text_b: str) -> str:
    return (
        "You are a strict semantic similarity scorer.\n"
        "Judge meaning, not surface form.\n"
        "Output only one decimal number between 0 and 1. Do not output anything else.\n\n"
        "Scoring rules:\n"
        "- Output 1.0 if the two strings have the same meaning, even if wording differs.\n"
        "- Output 0.0 if the two strings have different or contradictory meanings.\n"
        "- Use an intermediate value only when the meanings partially overlap.\n"
        "- Treat numerals and number words as equivalent when they denote the same value.\n"
        "- Treat normalized forms as equivalent when meaning is unchanged, such as case, punctuation, spacing, units, dates, or minor formatting.\n"
        f"String A:\n{text_a}\n\n"
        f"String B:\n{text_b}\n"
    )

def normalize_chat_url(api_base_url: str) -> str:
    url = api_base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"

def parse_similarity_score(text: str) -> float:
    match = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", text.strip())
    if match is None:
        raise ValueError(f"Failed to parse similarity score from response: {text!r}")
    score = float(match.group(1))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Similarity score out of range: {score}")
    return score

@function
def sglang_semantic_similarity(s, text_a: str, text_b: str):
    s += user(build_prompt(text_a, text_b))
    s += assistant(gen("score", max_tokens=8))

def semantic_similarity(
    text_a: str,
    text_b: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> float:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(text_a, text_b)}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(
        normalize_chat_url(api_base_url), json=payload, headers=headers, timeout=timeout
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    try:
        return parse_similarity_score(content)
    except ValueError:
        return 0.0

def semantic_similarity_strict(
    text_a: str,
    text_b: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> float:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(text_a, text_b)}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(
        normalize_chat_url(api_base_url), json=payload, headers=headers, timeout=timeout
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return parse_similarity_score(content)

def semantic_similarity_batch(
    text_pairs: list[tuple[str, str]],
    api_base_url: str = DEFAULT_API_BASE_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]:
    backend = RuntimeEndpoint(api_base_url.rstrip("/"))
    outputs: list[float] = []
    chunk_size = int(batch_size)
    for start in range(0, len(text_pairs), chunk_size):
        chunk = text_pairs[start : start + chunk_size]
        payloads = [{"text_a": text_a, "text_b": text_b} for text_a, text_b in chunk]
        batch_outputs = sglang_semantic_similarity.run_batch(payloads, backend=backend)
        for out in batch_outputs:
            try:
                raw_text = str(out["score"])
                outputs.append(parse_similarity_score(raw_text))
            except (KeyError, ValueError):
                outputs.append(0.0)
    return outputs

def semantic_similarity_batch_strict(
    text_pairs: list[tuple[str, str]],
    api_base_url: str = DEFAULT_API_BASE_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[float]:
    backend = RuntimeEndpoint(api_base_url.rstrip("/"))
    outputs: list[float] = []
    chunk_size = int(batch_size)
    for start in range(0, len(text_pairs), chunk_size):
        chunk = text_pairs[start : start + chunk_size]
        payloads = [{"text_a": text_a, "text_b": text_b} for text_a, text_b in chunk]
        batch_outputs = sglang_semantic_similarity.run_batch(payloads, backend=backend)
        for out in batch_outputs:
            try:
                raw_text = str(out["score"])
                outputs.append(parse_similarity_score(raw_text))
            except (KeyError, ValueError):
                outputs.append(0.0)
    return outputs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text_a", nargs="?")
    parser.add_argument("text_b", nargs="?")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        scores = semantic_similarity_batch(
            DEMO_PAIRS,
            api_base_url=args.api_base_url,
            batch_size=args.batch_size,
        )
        for (text_a, text_b), score in zip(DEMO_PAIRS, scores):
            print(f"{score}\t{text_a!r}\t{text_b!r}")
        return
    if args.text_a is None or args.text_b is None:
        raise ValueError("text_a and text_b are required unless --demo is used")
    score = semantic_similarity(
        text_a=args.text_a,
        text_b=args.text_b,
        api_base_url=args.api_base_url,
        model=args.model,
        timeout=args.timeout,
    )
    print(score)

if __name__ == "__main__":
    text_a = "Geoy, 14; Addsen, 12."
    text_b = "Addsen, 12, Geoy, 14."
    score = semantic_similarity(
        text_a=text_a,
        text_b=text_b,
        api_base_url=DEFAULT_API_BASE_URL,
        model=DEFAULT_MODEL,
        timeout=DEFAULT_TIMEOUT,
    )
    print(score)
