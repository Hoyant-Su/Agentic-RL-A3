import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4
import ray
import ray.actor
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
T = TypeVar("T")

class PoolMode(Enum):

    ThreadMode = 1
    ProcessMode = 2

@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count

class SearchExecutionWorker:

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = (
            self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        )

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(
            name="rate-limiter", get_if_exists=True
        ).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)

def init_search_execution_pool(
    num_workers: int,
    enable_global_rate_limit=True,
    rate_limit=10,
    mode: PoolMode = PoolMode.ThreadMode,
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(
                enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit
            )
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")

class SearchTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, (
            "Configuration must include 'retrieval_service_url'"
        )
        self.topk = config.get("topk", 3)
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")
        logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    def execute_search(
        self,
        instance_id: str,
        query_list: list,
        retrieval_service_url: str,
        topk: int,
        timeout: int,
    ):
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")
        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = (
                "Error: 'query_list' is missing, empty, or not a list in parameters."
            )
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"result": error_msg}), 0.0, {}
        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search,
                instance_id,
                query_list_from_params,
                self.retrieval_service_url,
                self.topk,
                timeout,
            )
            self._instance_dict[instance_id]["reward"].append(result_text.strip())
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }
            return result_text, 0.0, metrics
        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
