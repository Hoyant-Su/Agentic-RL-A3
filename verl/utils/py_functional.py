
import importlib
import multiprocessing
import os
import queue
import signal
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

def _mp_target_wrapper(
    target_func: Callable,
    mp_queue: multiprocessing.Queue,
    args: Tuple,
    kwargs: Dict[str, Any],
):
    try:
        result = target_func(*args, **kwargs)
        mp_queue.put((True, result))
    except Exception as e:
        try:
            import pickle

            pickle.dumps(e)
            mp_queue.put((False, e))
        except (pickle.PicklingError, TypeError):
            mp_queue.put(
                (
                    False,
                    RuntimeError(
                        f"Original exception type {type(e).__name__} not pickleable: {e}"
                    ),
                )
            )

def timeout_limit(seconds: float, use_signals: bool = False):

    def decorator(func):
        if use_signals:
            if os.name != "posix":
                raise NotImplementedError(f"Unsupported OS: {os.name}")
            print(
                "WARN: The 'use_signals=True' option in the timeout decorator is deprecated. \
                Signals are unreliable outside the main thread. \
                Please use the default multiprocessing-based timeout (use_signals=False)."
            )

            @wraps(func)
            def wrapper_signal(*args, **kwargs):
                def handler(signum, frame):
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {seconds} seconds (signal)!"
                    )

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result

            return wrapper_signal
        else:

            @wraps(func)
            def wrapper_mp(*args, **kwargs):
                q = multiprocessing.Queue(maxsize=1)
                process = multiprocessing.Process(
                    target=_mp_target_wrapper, args=(func, q, args, kwargs)
                )
                process.start()
                process.join(timeout=seconds)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive():
                        print(
                            f"Warning: Process {process.pid} did not terminate gracefully after timeout."
                        )
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {seconds} seconds (multiprocessing)!"
                    )
                try:
                    success, result_or_exc = q.get(timeout=0.1)
                    if success:
                        return result_or_exc
                    else:
                        raise result_or_exc
                except queue.Empty as err:
                    exitcode = process.exitcode
                    if exitcode is not None and exitcode != 0:
                        raise RuntimeError(
                            f"Child process exited with error (exitcode: {exitcode}) before returning result."
                        ) from err
                    else:
                        raise TimeoutError(
                            f"Operation timed out or process finished unexpectedly without result (exitcode: {exitcode})."
                        ) from err
                finally:
                    q.close()
                    q.join_thread()

            return wrapper_mp

    return decorator

def union_two_dict(dict1: Dict, dict2: Dict):
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], (
                f"{key} in meta_dict1 and meta_dict2 are not the same object"
            )
        dict1[key] = val
    return dict1

def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)

class NestedNamespace(SimpleNamespace):

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)

class DynamicEnumMeta(type):
    def __iter__(cls) -> Iterator[Any]:
        return iter(cls._registry.values())

    def __contains__(cls, item: Any) -> bool:
        if isinstance(item, str):
            return item in cls._registry
        return item in cls._registry.values()

    def __getitem__(cls, name: str) -> Any:
        return cls._registry[name]

    def __reduce_ex__(cls, protocol):
        return getattr, (importlib.import_module(cls.__module__), cls.__name__)

    def names(cls):
        return list(cls._registry.keys())

    def values(cls):
        return list(cls._registry.values())

class DynamicEnum(metaclass=DynamicEnumMeta):
    _registry: Dict[str, "DynamicEnum"] = {}
    _next_value: int = 0

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __reduce_ex__(self, protocol):
        module = importlib.import_module(self.__class__.__module__)
        enum_cls = getattr(module, self.__class__.__name__)
        return getattr, (enum_cls, self.name)

    @classmethod
    def register(cls, name: str) -> "DynamicEnum":
        key = name.upper()
        if key in cls._registry:
            raise ValueError(f"{key} already registered")
        member = cls(key, cls._next_value)
        cls._registry[key] = member
        setattr(cls, key, member)
        cls._next_value += 1
        return member

    @classmethod
    def remove(cls, name: str):
        key = name.upper()
        member = cls._registry.pop(key)
        delattr(cls, key)
        return member

    @classmethod
    def from_name(cls, name: str) -> Optional["DynamicEnum"]:
        return cls._registry.get(name.upper())

def convert_to_regular_types(obj):
    from omegaconf import ListConfig, DictConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return (
            {k: convert_to_regular_types(v) for k, v in obj.items()}
            if isinstance(obj, DictConfig)
            else list(obj)
        )
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj
