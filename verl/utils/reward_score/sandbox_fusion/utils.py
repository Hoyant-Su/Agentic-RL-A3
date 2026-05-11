import concurrent.futures
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple
import requests

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10
logger = logging.getLogger(__name__)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]

def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: str,
    compile_timeout: int,
    run_timeout: int,
    language: str = "python",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    request_id = str(uuid.uuid4())
    log_prefix = f"[Request ID: {request_id}] "
    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg
    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "language": language,
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}"
            )
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,
            )
            if response.status_code == 504:
                last_error = f"{log_prefix}API Request Error: Gateway Timeout (504) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue
            response.raise_for_status()
            logger.info(
                f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}"
            )
            return response.json(), None
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")
    return None, last_error.replace(
        log_prefix, "API Call Failed: "
    ) if last_error else "API Call Failed after retries"

def _process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    api_response = None
    error_msg = None
    logger.info(f"Processing test case {case_index + 1}.")
    current_generation_code = generation
    if fn_name and language == "python":
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
# === User's Original Code START ===
{generation}
# === User's Original Code END ===
_SANDBOX_FN_NAME = "{fn_name}"
def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: {{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred
    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class() 
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)
        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred
        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred
if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()
    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # Optional: To explicitly exit with an error code if the sandbox relies on it
    # else:
    #    sys.exit(1) 
"""
        current_generation_code = wrapper_code
    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    code=current_generation_code,
                    stdin=str(stdin_data),
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    language=language,
                )
        else:
            api_response, error_msg = call_sandbox_api(
                sandbox_fusion_url=sandbox_fusion_url,
                code=current_generation_code,
                stdin=str(stdin_data),
                compile_timeout=timeout,
                run_timeout=timeout,
                language=language,
            )
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        logger.error(f"Case {case_index + 1}: {error_msg}")
        traceback.print_exc()
    metadata = {
        "case_index": case_index,
        "input": str(stdin_data),
        "expected_output": str(expected_output),
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    result_status = -1
    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1
        logger.error(f"Case {case_index}: API error occurred: {error_msg}")
        generation_to_log = (
            generation[:200] + "..." if len(generation) > 200 else generation
        )
        logger.error(f"Case {case_index}: code: {generation_to_log}")
        logger.error(f"Case {case_index}: input: {str(stdin_data)}")
    elif api_response:
        logger.debug(f"Case {case_index}: API Response: {api_response}")
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")
        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")
        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")
        api_status = metadata["api_status"]
        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1
        elif api_status == "Failed":
            logger.debug(f"API returned Failed status. Response: {api_response}")
            logger.debug(f"Compile Result: {compile_result}")
            logger.debug(f"Run Result: {run_result}")
            is_compile_error = compile_result and (
                metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
                or (
                    metadata["compile_status"] == "Finished"
                    and compile_result.get("return_code") != 0
                )
            )
            if is_compile_error:
                if metadata["compile_status"] == "TimeLimitExceeded":
                    metadata["status"] = "compile_timeout"
                else:
                    metadata["status"] = "compile_error"
                result_status = -4
            elif run_result:
                is_runtime_error = (
                    metadata["run_status"] == "TimeLimitExceeded"
                    or metadata["run_status"] == "Error"
                    or (
                        metadata["run_status"] == "Finished"
                        and run_result.get("return_code") != 0
                    )
                )
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"
                        result_status = -3
                    else:
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    logger.warning(
                        f"Unknown run_status '{metadata['run_status']}' or state within Failed API status."
                    )
                    metadata["status"] = "unknown_failure"
                    result_status = -1
            else:
                logger.warning(
                    "API status Failed but cannot determine specific error type (compile/run)."
                )
                metadata["status"] = "unknown_failure_state"
                result_status = -1
        elif api_status == "Success":
            if run_result and metadata["run_status"] == "Finished":
                actual_output = (
                    metadata["stdout"] if metadata["stdout"] is not None else ""
                )
                if str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                    result_status = True
                    metadata["status"] = "success"
                else:
                    result_status = False
                    metadata["status"] = "wrong_answer"
            else:
                metadata["status"] = "unexpected_success_state"
                result_status = -1
        else:
            logger.warning(f"Unknown API status received: {api_status}")
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1
    else:
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(
            f"Case {case_index}: Unknown API state (no response and no error message)."
        )
    return result_status, metadata

def check_correctness(
    sandbox_fusion_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    logger.info("Starting correctness check for generation.")
    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]
    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    results = [None] * num_cases
    metadata_list = [None] * num_cases
    if num_cases == 0:
        logger.warning("Empty inputs provided.")
        return [], []
    if len(inputs) != len(expected_outputs):
        logger.warning(
            f"Mismatch between number of inputs ({len(inputs)}) and outputs ({len(expected_outputs)})."
        )
        return [-1] * num_cases, [
            {"error": "Input/output count mismatch", "case_index": i}
            for i in range(num_cases)
        ]
    first_compile_error_index = -1
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(32, os.cpu_count() * 5)
    ) as executor:
        future_to_index = {
            executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                sandbox_fusion_url,
                generation,
                timeout,
                language,
                concurrent_semaphore,
                fn_name,
            ): i
            for i, stdin_data in enumerate(inputs)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata
                if result_status == -4:
                    if (
                        first_compile_error_index == -1
                        or index < first_compile_error_index
                    ):
                        first_compile_error_index = index
            except Exception as exc:
                logger.error(f"Test case {index} generated an exception: {exc}")
                traceback.print_exc()
                results[index] = -1
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]),
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }
    if first_compile_error_index != -1:
        logger.warning(
            f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors."
        )
        for i in range(first_compile_error_index + 1, num_cases):
            if results[i] != -4:
                results[i] = -4
                if metadata_list[i] is None:
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]),
                        "api_request_error": None,
                        "status": "compile_error_skipped",
                    }
                else:
                    metadata_list[i]["status"] = "compile_error_skipped"
    logger.info(f"Correctness check finished. Results: {results}")
    return results, metadata_list
