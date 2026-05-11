import json
from typing import Any, Literal
from pydantic import BaseModel

class OpenAIFunctionPropertySchema(BaseModel):

    type: str
    description: str | None = None
    enum: list[str] | None = None

class OpenAIFunctionParametersSchema(BaseModel):

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]

class OpenAIFunctionSchema(BaseModel):

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False

class OpenAIFunctionToolSchema(BaseModel):

    type: str
    function: OpenAIFunctionSchema

class OpenAIFunctionParsedSchema(BaseModel):

    name: str
    arguments: str

class OpenAIFunctionCallSchema(BaseModel):

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True
        return OpenAIFunctionCallSchema(
            name=parsed_schema.name, arguments=arguments
        ), has_decode_error

class OpenAIFunctionToolCall(BaseModel):

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema
