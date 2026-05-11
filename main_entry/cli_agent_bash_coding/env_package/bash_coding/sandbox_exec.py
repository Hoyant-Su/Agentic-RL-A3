
from __future__ import annotations
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import ctypes

@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    returncode: int

class SandboxError(RuntimeError):
    pass

def _which(cmd: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, cmd)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

_SYS_landlock_create_ruleset = 444
_SYS_landlock_add_rule = 445
_SYS_landlock_restrict_self = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LL_FS_EXECUTE = 1 << 0
_LL_FS_WRITE_FILE = 1 << 1
_LL_FS_READ_FILE = 1 << 2
_LL_FS_READ_DIR = 1 << 3
_LL_FS_REMOVE_DIR = 1 << 4
_LL_FS_REMOVE_FILE = 1 << 5
_LL_FS_MAKE_CHAR = 1 << 6
_LL_FS_MAKE_DIR = 1 << 7
_LL_FS_MAKE_REG = 1 << 8
_LL_FS_MAKE_SOCK = 1 << 9
_LL_FS_MAKE_FIFO = 1 << 10
_LL_FS_MAKE_BLOCK = 1 << 11
_LL_FS_MAKE_SYM = 1 << 12
_LL_FS_REFER = 1 << 13

def _handled_fs_for_abi(abi: int) -> int:
    handled = (
        _LL_FS_EXECUTE
        | _LL_FS_WRITE_FILE
        | _LL_FS_READ_FILE
        | _LL_FS_READ_DIR
        | _LL_FS_REMOVE_DIR
        | _LL_FS_REMOVE_FILE
        | _LL_FS_MAKE_CHAR
        | _LL_FS_MAKE_DIR
        | _LL_FS_MAKE_REG
        | _LL_FS_MAKE_SOCK
        | _LL_FS_MAKE_FIFO
        | _LL_FS_MAKE_BLOCK
        | _LL_FS_MAKE_SYM
    )
    if abi >= 2:
        handled |= _LL_FS_REFER
    return handled

class _landlock_ruleset_attr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]

class _landlock_path_beneath_attr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

def _landlock_abi_version() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    res = libc.syscall(
        _SYS_landlock_create_ruleset, 0, 0, _LANDLOCK_CREATE_RULESET_VERSION
    )
    if res < 0:
        err = ctypes.get_errno()
        raise SandboxError(f"Landlock not available (errno={err}: {os.strerror(err)})")
    return int(res)

def _prctl_no_new_privs() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    res = libc.prctl(38, 1, 0, 0, 0)
    if res != 0:
        err = ctypes.get_errno()
        raise SandboxError(
            f"Failed to set no_new_privs (errno={err}: {os.strerror(err)})"
        )

def _open_path_fd(path: str) -> int:
    o_path = getattr(os, "O_PATH", 0x200000)
    return os.open(path, o_path | os.O_CLOEXEC)

def _apply_landlock_restriction(*, work_dir: str, ro_paths: Iterable[str]) -> None:
    abi = _landlock_abi_version()
    _prctl_no_new_privs()
    libc = ctypes.CDLL(None, use_errno=True)
    ruleset_attr = _landlock_ruleset_attr(handled_access_fs=_handled_fs_for_abi(abi))
    ruleset_fd = libc.syscall(
        _SYS_landlock_create_ruleset,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        err = ctypes.get_errno()
        raise SandboxError(
            f"landlock_create_ruleset failed (errno={err}: {os.strerror(err)})"
        )

    def _add_path_rule(path: str, allowed: int) -> None:
        fd = _open_path_fd(path)
        try:
            attr = _landlock_path_beneath_attr(allowed_access=allowed, parent_fd=fd)
            res = libc.syscall(
                _SYS_landlock_add_rule,
                ruleset_fd,
                _LANDLOCK_RULE_PATH_BENEATH,
                ctypes.byref(attr),
                0,
            )
            if res != 0:
                err = ctypes.get_errno()
                raise SandboxError(
                    f"landlock_add_rule({path}) failed (errno={err}: {os.strerror(err)})"
                )
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    ro_allowed = _LL_FS_READ_FILE | _LL_FS_READ_DIR | _LL_FS_EXECUTE
    for p in ro_paths:
        if os.path.exists(p):
            _add_path_rule(p, ro_allowed)
    rw_allowed = (
        _LL_FS_EXECUTE
        | _LL_FS_READ_FILE
        | _LL_FS_READ_DIR
        | _LL_FS_WRITE_FILE
        | _LL_FS_REMOVE_DIR
        | _LL_FS_REMOVE_FILE
        | _LL_FS_MAKE_DIR
        | _LL_FS_MAKE_REG
        | _LL_FS_MAKE_SYM
    )
    if abi >= 2:
        rw_allowed |= _LL_FS_REFER
    _add_path_rule(work_dir, rw_allowed)
    res = libc.syscall(_SYS_landlock_restrict_self, ruleset_fd, 0)
    if res != 0:
        err = ctypes.get_errno()
        raise SandboxError(
            f"landlock_restrict_self failed (errno={err}: {os.strerror(err)})"
        )
    try:
        os.close(ruleset_fd)
    except OSError:
        pass

def _validate_command(command: str) -> None:
    cmd = command.strip()
    if not cmd:
        raise SandboxError("Empty command is not allowed.")
    lowered = " ".join(cmd.split()).lower()
    if lowered.startswith("rm ") and (
        " -rf" in lowered
        or " -fr" in lowered
        or " -r " in lowered
        or lowered.startswith("rm -r")
    ):
        raise SandboxError("Recursive delete (rm -r/-rf) is blocked by sandbox policy.")
    fork_bomb_patterns = [
        ":(){",
        ":() {",
        "|:&",
        "| :&",
        "$0 &",
        " :|: ",
        " : | : ",
    ]
    cmd_compact = " ".join(cmd.split())
    for pat in fork_bomb_patterns:
        if pat in cmd_compact:
            raise SandboxError(
                f"Fork bomb or unbounded process creation pattern detected "
                f"(blocked: {pat!r}). Sandbox policy forbids recursive/background fork patterns."
            )

def run_in_unshare_sandbox(
    *,
    work_dir: str,
    command: str,
    timeout_s: int = 10,
    extra_env: Dict[str, str] | None = None,
) -> ExecResult:
    _validate_command(command)
    if not os.path.isabs(work_dir):
        work_dir = os.path.abspath(work_dir)
    if not os.path.isdir(work_dir):
        raise SandboxError(f"work_dir does not exist: {work_dir}")
    unshare_path = _which("unshare")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env["HOME"] = os.path.join(work_dir, "home")
    env["TMPDIR"] = os.path.join(work_dir, ".sandbox", "tmp")
    os.makedirs(env["HOME"], exist_ok=True)
    os.makedirs(env["TMPDIR"], exist_ok=True)
    env["TERM"] = "dumb"
    env["NONINTERACTIVE"] = "1"
    env["DEBIAN_FRONTEND"] = "noninteractive"
    env["CI"] = "true"
    ro_paths = ["/bin", "/usr", "/lib", "/lib64", "/etc"]
    safe_command = f"{{ {command}; }} < /dev/null"
    launcher = f"""
import os, ctypes
work_dir = {work_dir!r}
ro_paths = {ro_paths!r}
cmd = {safe_command!r}
SYS_landlock_create_ruleset = 444
SYS_landlock_add_rule = 445
SYS_landlock_restrict_self = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
LL_EXECUTE = 1 << 0
LL_WRITE_FILE = 1 << 1
LL_READ_FILE = 1 << 2
LL_READ_DIR = 1 << 3
LL_REMOVE_DIR = 1 << 4
LL_REMOVE_FILE = 1 << 5
LL_MAKE_CHAR = 1 << 6
LL_MAKE_DIR = 1 << 7
LL_MAKE_REG = 1 << 8
LL_MAKE_SOCK = 1 << 9
LL_MAKE_FIFO = 1 << 10
LL_MAKE_BLOCK = 1 << 11
LL_MAKE_SYM = 1 << 12
LL_REFER = 1 << 13
libc = ctypes.CDLL(None, use_errno=True)
# no_new_privs (required by Landlock)
PR_SET_NO_NEW_PRIVS = 38
if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
    e = ctypes.get_errno()
    raise SystemExit(f'no_new_privs failed: errno={{e}} {{os.strerror(e)}}')
# Probe ABI version
abi = libc.syscall(SYS_landlock_create_ruleset, 0, 0, LANDLOCK_CREATE_RULESET_VERSION)
if abi < 0:
    e = ctypes.get_errno()
    raise SystemExit(f'landlock not available: errno={{e}} {{os.strerror(e)}}')
handled = (LL_EXECUTE|LL_WRITE_FILE|LL_READ_FILE|LL_READ_DIR|LL_REMOVE_DIR|LL_REMOVE_FILE|LL_MAKE_CHAR|LL_MAKE_DIR|LL_MAKE_REG|LL_MAKE_SOCK|LL_MAKE_FIFO|LL_MAKE_BLOCK|LL_MAKE_SYM)
if abi >= 2:
    handled |= LL_REFER
class ruleset_attr(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]
class path_beneath_attr(ctypes.Structure):
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int)]
rs_attr = ruleset_attr(handled_access_fs=handled)
rs_fd = libc.syscall(SYS_landlock_create_ruleset, ctypes.byref(rs_attr), ctypes.sizeof(rs_attr), 0)
if rs_fd < 0:
    e = ctypes.get_errno()
    raise SystemExit(f'create_ruleset failed: errno={{e}} {{os.strerror(e)}}')
O_PATH = getattr(os, 'O_PATH', 0x200000)
def open_path(p: str) -> int:
    return os.open(p, O_PATH | os.O_CLOEXEC)
def add_rule(path: str, allowed: int) -> None:
    fd = open_path(path)
    try:
        attr = path_beneath_attr(allowed_access=allowed, parent_fd=fd)
        r = libc.syscall(SYS_landlock_add_rule, rs_fd, LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(attr), 0)
        if r != 0:
            e = ctypes.get_errno()
            raise SystemExit(f'add_rule({{path}}) failed: errno={{e}} {{os.strerror(e)}}')
    finally:
        try: os.close(fd)\n        except OSError: pass
# Allow runtime reads/exec for system dirs
ro_allowed = (LL_READ_FILE | LL_READ_DIR | LL_EXECUTE)
for p in ro_paths:
    if os.path.exists(p):
        add_rule(p, ro_allowed)
# Allow /proc read (helps tools behave locally); no writes.
if os.path.exists('/proc'):
    add_rule('/proc', (LL_READ_FILE | LL_READ_DIR))
# Allow minimal /dev devices (read/write)
for dev in ['/dev/null', '/dev/zero', '/dev/urandom', '/dev/random']:
    if os.path.exists(dev):
        add_rule(dev, (LL_READ_FILE | LL_WRITE_FILE))
# Allow work_dir RW
rw = (LL_EXECUTE|LL_READ_FILE|LL_READ_DIR|LL_WRITE_FILE|LL_REMOVE_DIR|LL_REMOVE_FILE|LL_MAKE_DIR|LL_MAKE_REG|LL_MAKE_SYM)
if abi >= 2:
    rw |= LL_REFER
add_rule(work_dir, rw)
r = libc.syscall(SYS_landlock_restrict_self, rs_fd, 0)
if r != 0:
    e = ctypes.get_errno()
    raise SystemExit(f'restrict_self failed: errno={{e}} {{os.strerror(e)}}')
try: os.close(rs_fd)\nexcept OSError: pass
os.chdir(work_dir)
# Set non-interactive environment in the launcher as well
os.environ['TERM'] = 'dumb'
os.environ['NONINTERACTIVE'] = '1'
os.environ['DEBIAN_FRONTEND'] = 'noninteractive'
os.environ['CI'] = 'true'
os.execv('/bin/bash', ['bash', '-lc', cmd])
"""
    if unshare_path:
        cmd = [unshare_path, "--map-root-user", "-Urn", sys.executable, "-c", launcher]
    else:
        cmd = [sys.executable, "-c", launcher]
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=work_dir,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        stdout, stderr = p.communicate(timeout=timeout_s)
        proc = subprocess.CompletedProcess(
            cmd, returncode=p.returncode, stdout=stdout, stderr=stderr
        )
        return ExecResult(
            stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode
        )
    except subprocess.TimeoutExpired as e:
        import signal

        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                p.kill()
            except OSError:
                pass
        try:
            sid = os.getsid(p.pid)
            subprocess.run(
                ["pkill", "-9", "-s", str(sid)],
                timeout=2,
                capture_output=True,
            )
        except (ProcessLookupError, FileNotFoundError, OSError):
            pass
        raise SandboxError(f"Sandbox execution timed out after {timeout_s}s") from e
