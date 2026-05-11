
import asyncio
import concurrent.futures
import os
from typing import Any, Optional
import ray

def ray_noset_visible_devices(env_vars=os.environ):
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)

def parallel_put(data_list: list[Any], max_workers: Optional[int] = None):
    assert len(data_list) > 0, "data_list must not be empty"

    def put_data(index, data):
        return index, ray.put(data)

    if max_workers is None:
        max_workers = min(len(data_list), 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [
            executor.submit(put_data, i, data) for i, data in enumerate(data_list)
        ]
        res_lst = []
        for future in concurrent.futures.as_completed(data_list_f):
            res_lst.append(future.result())
        output = [None for _ in range(len(data_list))]
        for res in res_lst:
            index, data_ref = res
            output[index] = data_ref
    return output

def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
