from typing import Any, Optional, Tuple
from uuid import uuid4
from .schemas import OpenAIFunctionToolSchema

class BaseTool:

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.name = tool_schema.function.name
        self.tool_schema = tool_schema

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        return "Updated the tool state.", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass
