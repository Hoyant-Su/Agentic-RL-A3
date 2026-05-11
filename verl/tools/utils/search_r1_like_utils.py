import json
import logging
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple
import requests

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10
logger = logging.getLogger(__name__)

def call_search_api(
    retrieval_service_url: str,
    query_list: List[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "
    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue
            response.raise_for_status()
            logger.info(
                f"{log_prefix}Search API call successful on attempt {attempt + 1}"
            )
            return response.json(), None
        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(
        log_prefix, "API Call Failed: "
    ) if last_error else "API Call Failed after retries"

def _passages2string(retrieval_result):
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()

def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: List[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[str, Dict[str, Any]]:
    logger.info(f"Starting batch search for {len(query_list)} queries.")
    api_response = None
    error_msg = None
    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_msg}")
        traceback.print_exc()
    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }
    result_text = json.dumps(
        {"result": "Search request failed or timed out after retries."}
    )
    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Search error: {error_msg}"})
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")
        metadata["api_response"] = api_response
        try:
            raw_results = api_response.get("result", [])
            if raw_results:
                pretty_results = []
                total_results = 0
                for retrieval in raw_results:
                    formatted = _passages2string(retrieval)
                    pretty_results.append(formatted)
                    total_results += (
                        len(retrieval) if isinstance(retrieval, list) else 1
                    )
                final_result = "\n---\n".join(pretty_results)
                result_text = json.dumps({"result": final_result})
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
                logger.info(
                    f"Batch search: Successful, got {total_results} total results"
                )
            else:
                result_text = json.dumps({"result": "No search results found."})
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.info("Batch search: No results found")
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            result_text = json.dumps({"result": error_msg})
            metadata["status"] = "processing_error"
            logger.error(f"Batch search: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}
        )
        logger.error("Batch search: Unknown API state.")
    return result_text, metadata
