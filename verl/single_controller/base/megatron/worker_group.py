from typing import Dict
from verl.single_controller.base import ResourcePool, WorkerGroup
from .worker import DistGlobalInfo, DistRankInfo

class MegatronWorkerGroup(WorkerGroup):
    def __init__(self, resource_pool: ResourcePool, **kwargs):
        super().__init__(resource_pool=resource_pool, **kwargs)
        self._megatron_rank_info = None
        self._megatron_global_info: DistGlobalInfo = None

    def init_megatron(self, default_megatron_kwargs: Dict = None):
        raise NotImplementedError(
            "MegatronWorkerGroup.init_megatron should be overwritten"
        )

    def get_megatron_rank_info(self, rank: int) -> DistRankInfo:
        assert 0 <= rank < self.world_size, (
            f"rank must be from [0, world_size), Got {rank}"
        )
        return self._megatron_rank_info[rank]

    @property
    def tp_size(self):
        assert self._megatron_global_info is not None, (
            "MegatronWorkerGroup._megatron_global_info must be initialized"
        )
        return self._megatron_global_info.tp_size

    @property
    def dp_size(self):
        assert self._megatron_global_info is not None, (
            "MegatronWorkerGroup._megatron_global_info must be initialized"
        )
        return self._megatron_global_info.dp_size

    @property
    def pp_size(self):
        assert self._megatron_global_info is not None, (
            "MegatronWorkerGroup._megatron_global_info must be initialized"
        )
        return self._megatron_global_info.pp_size

    @property
    def cp_size(self):
        assert self._megatron_global_info is not None, (
            "MegatronWorkerGroup._megatron_global_info must be initialized"
        )
        return self._megatron_global_info.cp_size

    def get_megatron_global_info(self):
        return self._megatron_global_info
