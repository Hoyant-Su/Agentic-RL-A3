from importlib import import_module
from typing import Any
from omegaconf import OmegaConf

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

METHOD_REGISTRY = {
    "commit_credit": (
        "main_entry.cli_agent_bash_coding.baseline_methods.commit_credit.runtime",
        "CommitCreditRuntime",
    ),
    "retroagent": (
        "main_entry.cli_agent_bash_coding.baseline_methods.retroagent_lite.runtime",
        "RetroAgentRuntime",
    ),
}

class BaselineMethodManager:
    def __init__(self, methods: dict[str, Any]) -> None:
        self.methods = methods

    @classmethod
    def from_config(cls, config) -> "BaselineMethodManager":
        methods: dict[str, Any] = {}
        for method_name, (module_path, class_name) in METHOD_REGISTRY.items():
            enable_value = OmegaConf.select(
                config, f"reward.{method_name}_enable", default=None
            )
            if enable_value is None or not _as_bool(enable_value):
                continue
            module = import_module(module_path)
            method_cls = getattr(module, class_name)
            methods[method_name] = method_cls.from_config(config)
        return cls(methods)

    def enabled(self, method_name: str | None = None) -> bool:
        if method_name is None:
            return bool(self.methods)
        method = self.methods.get(method_name)
        if method is None:
            return False
        is_enabled = getattr(method, "is_enabled", None)
        if callable(is_enabled):
            return bool(is_enabled())
        return True

    def prepare_batch(self, tasks: list[str], *, is_eval: bool, group_n: int) -> None:
        for method in self.methods.values():
            prepare_batch = getattr(method, "prepare_batch", None)
            if callable(prepare_batch):
                prepare_batch(tasks, is_eval=is_eval, group_n=group_n)

    def build_prompt_context(self, index: int) -> str:
        contexts: list[str] = []
        for method in self.methods.values():
            build_prompt_context = getattr(method, "build_prompt_context", None)
            if not callable(build_prompt_context):
                continue
            context = build_prompt_context(index)
            if context:
                contexts.append(str(context))
        return "\n\n".join(contexts)

    def apply_post_rollout(self, **kwargs) -> dict[str, list[dict[str, Any]]]:
        outputs: dict[str, list[dict[str, Any]]] = {}
        for method_name, method in self.methods.items():
            apply_post_rollout = getattr(method, "apply_post_rollout", None)
            if not callable(apply_post_rollout):
                continue
            result = apply_post_rollout(**kwargs)
            if result is not None:
                outputs[method_name] = result
        return outputs
