import json
import os
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0",
    },
}

def get_ppo_ray_runtime_env():
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}"))
        .get("runtime_env", {})
        .get("working_dir", None)
    )
    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
