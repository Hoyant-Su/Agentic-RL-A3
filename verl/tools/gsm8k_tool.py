import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4
from verl.utils.reward_score import gsm8k
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class Gsm8kTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        if answer.startswith("#### "):
            self._instance_dict[instance_id]["response"] = answer
        else:
            self._instance_dict[instance_id]["response"] = "#### " + answer
        reward = await self.calc_reward(instance_id)
        tool_reward = (
            0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        )
        self._instance_dict[instance_id]["reward"] = reward
        return f"Current parsed {answer=} {reward=}", tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
