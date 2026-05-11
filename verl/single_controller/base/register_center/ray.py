from typing import Dict
import ray

@ray.remote
class WorkerGroupRegisterCenter:
    def __init__(self, rank_zero_info):
        self.rank_zero_info = rank_zero_info
        self.workers_info: Dict[int, str] = {}

    def get_rank_zero_info(self):
        return self.rank_zero_info

    def set_worker_info(self, rank, node_id) -> None:
        self.workers_info[rank] = node_id

    def get_worker_info(self) -> Dict[int, str]:
        return self.workers_info

def create_worker_group_register_center(name, info):
    return WorkerGroupRegisterCenter.options(name=name).remote(info)
