from importlib.metadata import PackageNotFoundError, version
from packaging import version as vs
from verl.utils.import_utils import is_sglang_available

def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None

package_name = "vllm"
package_version = get_version(package_name)
vllm_version = None
if package_version == "0.5.4":
    vllm_version = "0.5.4"
    from .vllm_v_0_5_4 import parallel_state
    from .vllm_v_0_5_4.llm import LLM, LLMEngine
elif package_version == "0.6.3" or package_version.startswith("0.6.3"):
    vllm_version = "0.6.3"
    from .vllm_v_0_6_3 import parallel_state
    from .vllm_v_0_6_3.llm import LLM, LLMEngine
elif vs.parse(package_version) >= vs.parse("0.7.0"):
    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported vllm versions are 0.6.3 and 0.7.0+"
        )
__all__ = ["LLM", "LLMEngine", "parallel_state"]
