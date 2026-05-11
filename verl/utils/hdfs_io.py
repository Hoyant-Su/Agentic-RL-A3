import logging
import os
import shutil

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))
_HDFS_PREFIX = "hdfs://"
_HDFS_BIN_PATH = shutil.which("hdfs")

def exists(path: str, **kwargs) -> bool:
    if _is_non_local(path):
        return _exists(path, **kwargs)
    return os.path.exists(path)

def _exists(file_path: str):
    if file_path.startswith("hdfs"):
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    return os.path.exists(file_path)

def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    if _is_non_local(name):
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)

def _mkdir(file_path: str) -> bool:
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True

def copy(src: str, dst: str, **kwargs) -> bool:
    if _is_non_local(src) or _is_non_local(dst):
        return _copy(src, dst)
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)

def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(
                _hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout
            )
        else:
            returncode = _run_cmd(
                _hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout
            )
    else:
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(
                _hdfs_cmd(
                    f"-get \
                {from_path} {to_path}"
                ),
                timeout=timeout,
            )
        else:
            try:
                shutil.copy(from_path, to_path)
                returncode = 0
            except shutil.SameFileError:
                returncode = 0
            except Exception as e:
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                returncode = -1
    return returncode == 0

def _run_cmd(cmd: str, timeout=None):
    return os.system(cmd)

def _hdfs_cmd(cmd: str) -> str:
    return f"{_HDFS_BIN_PATH} dfs {cmd}"

def _is_non_local(path: str):
    return path.startswith(_HDFS_PREFIX)
