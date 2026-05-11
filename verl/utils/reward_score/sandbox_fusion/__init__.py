import json
import logging
import traceback
from .utils import check_correctness

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)

def compute_score(
    sandbox_fusion_url,
    concurrent_semaphore,
    completion,
    test_cases,
    continuous=False,
    timeout=10,
):
    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():
                    solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]
    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]
        if not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            logger.error("Invalid test_cases structure.")
            return 0.0, [
                {"error": "Invalid test_cases structure (missing inputs/outputs)"}
            ]
        res_list, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_fusion_url,
            in_outs=test_cases,
            generation=solution,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
        )
        if not res_list:
            return 0.0, metadata_list
        if continuous:
            num_to_consider = min(len(res_list), 10)
            if num_to_consider == 0:
                score = 0.0
            else:
                passed_count = sum(1 for r in res_list[:num_to_consider] if r is True)
                score = passed_count / num_to_consider
            final_metadata = metadata_list
        else:
            passed_count = sum(1 for r in res_list if r is True)
            total_cases = len(res_list)
            score = passed_count / total_cases if total_cases > 0 else 0.0
            final_metadata = metadata_list
    except Exception as e:
        logger.error(f"Error during compute_score: {e}")
        traceback.print_exc()
        score = 0.0
        final_metadata = (
            metadata_list
            if "metadata_list" in locals()
            else [{"error": f"Unhandled exception: {e}"}]
        )
    return float(score), final_metadata if isinstance(final_metadata, list) else [
        final_metadata
    ]
