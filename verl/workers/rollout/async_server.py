import asyncio
import heapq
import importlib
import logging
import os
import socket
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Tuple, Type
from uuid import uuid4
import aiohttp
import fastapi
import ray
import uvicorn
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

class AsyncServerBase(ABC):

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print("FastAPI startup")
            self.server_ready.set()
            yield
            print(
                "FastAPI shutdown, maybe address already in use, exit process immediately."
            )
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route(
            "/v1/chat/completions", self.chat_completion, methods=["POST"]
        )
        self.port = _get_free_port()
        config = uvicorn.Config(
            app, host=["::", "0.0.0.0"], port=self.port, log_level="warning"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> Tuple[str, int]:
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    @abstractmethod
    async def chat_completion(self, raw_request: Request):
        raise NotImplementedError

    @abstractmethod
    async def init_engine(self):
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self):
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        raise NotImplementedError

class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        self.config = config
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

    async def submit_chat_completions(
        self,
        callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None],
        callback_additional_info: Dict[str, Any],
        **chat_complete_request,
    ):
        if "extra_headers" not in chat_complete_request:
            chat_complete_request["extra_headers"] = {}
        extra_headers = chat_complete_request["extra_headers"]
        request_id = extra_headers.get("x-request-id", None)
        if request_id:
            if request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
                extra_headers["x-request-id"] = request_id
            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address
        chat_complete_request["extra_headers"]["x-request-id"] = request_id
        completions, exception = None, None
        try:
            completions = await self._chat_completions_aiohttp(
                address, **chat_complete_request
            )
        except Exception as e:
            exception = e
        await callback(completions, callback_additional_info, exception)

    async def _chat_completions_openai(
        self, address: str, **chat_complete_request
    ) -> ChatCompletion:
        client = AsyncOpenAI(
            base_url=f"http://{address}/v1",
            api_key="token-abc123",
            timeout=None,
            max_retries=0,
        )
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(
        self, address: str, **chat_complete_request
    ) -> ChatCompletion:
        try:
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(
        self, prompts: DataProto, **sampling_params
    ) -> DataProto:
        raise NotImplementedError

class AsyncLLMServerManager:

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup,
        *,
        scheduler_kwargs: Dict[str, Any] = None,
    ):
        self.config = config
        self.worker_group = worker_group
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size
        register_center = ray.get_actor(
            f"{self.worker_group.name_prefix}_register_center"
        )
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size
        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size
        server_class = async_server_class(
            rollout_backend=self.config.rollout.name,
        )
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(
                    config,
                    self.rollout_dp_size,
                    rollout_dp_rank,
                    self.worker_group.name_prefix,
                )
                for rollout_dp_rank in unready_dp_ranks
            }
            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(
                        f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting..."
                    )
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])
        self.chat_scheduler: ChatCompletionScheduler = None
        self.chat_scheduler_loop = None
        self.chat_scheduler_ready = threading.Event()
        self.chat_scheduler_thread = threading.Thread(
            target=self._init_chat_scheduler, daemon=True
        )
        self.chat_scheduler_thread.start()
        self.chat_scheduler_ready.wait()

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)
        module_path, class_name = self.config.rollout.chat_scheduler.rsplit(".", 1)
        module = importlib.import_module(module_path)
        scheduler_cls = getattr(module, class_name)
        self.chat_scheduler = scheduler_cls(
            config=self.config.rollout,
            model_path=self.config.model.path,
            server_addresses=self.server_addresses,
            **self.scheduler_kwargs,
        )
        self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()

    def wake_up(self):
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def submit_chat_completions(
        self,
        callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None],
        callback_additional_info: Dict[str, Any],
        **chat_complete_request,
    ):
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info=callback_additional_info,
                **chat_complete_request,
            ),
            self.chat_scheduler_loop,
        )
        future.result()

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler.generate_sequences(prompts, **sampling_params),
            self.chat_scheduler_loop,
        )
        return future.result()

def async_server_class(rollout_backend: str) -> Type[AsyncServerBase]:
    if rollout_backend == "vllm":
        from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

        return AsyncvLLMServer
    elif rollout_backend == "sglang":
        from verl.workers.rollout.sglang_rollout.async_sglang_server import (
            AsyncSglangServer,
        )

        return AsyncSglangServer
    else:
        raise NotImplementedError
