#!/usr/bin/env python

import hashlib
import os
import shutil
import tempfile
import hashlib

try:
    from hdfs_io import copy, exists, makedirs
except ImportError:
    from .hdfs_io import copy, exists, makedirs
__all__ = ["copy", "exists", "makedirs"]
_HDFS_PREFIX = "hdfs://"

def is_non_local(path):
    return path.startswith(_HDFS_PREFIX)

def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()

def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    return dst

def verify_copy(src: str, dest: str) -> bool:
    if not os.path.exists(src):
        return False
    if not os.path.exists(dest):
        return False
    if os.path.isfile(src) != os.path.isfile(dest):
        return False
    if os.path.isfile(src):
        src_size = os.path.getsize(src)
        dest_size = os.path.getsize(dest)
        if src_size != dest_size:
            return False
        return True
    src_files = set()
    dest_files = set()
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_root = os.path.join(dest, rel_path) if rel_path != "." else dest
        if not os.path.exists(dest_root):
            return False
        for entry in os.listdir(root):
            src_entry = os.path.join(root, entry)
            src_files.add(os.path.relpath(src_entry, src))
        for entry in os.listdir(dest_root):
            dest_entry = os.path.join(dest_root, entry)
            dest_files.add(os.path.relpath(dest_entry, dest))
    if src_files != dest_files:
        return False
    for rel_path in src_files:
        src_entry = os.path.join(src, rel_path)
        dest_entry = os.path.join(dest, rel_path)
        if os.path.isdir(src_entry) != os.path.isdir(dest_entry):
            return False
        if os.path.isfile(src_entry):
            src_size = os.path.getsize(src_entry)
            dest_size = os.path.getsize(dest_entry)
            if src_size != dest_size:
                return False
    return True

def copy_to_shm(src: str):
    shm_model_root = "/dev/shm/verl-cache/"
    src_abs = os.path.abspath(os.path.normpath(src))
    dest = os.path.join(
        shm_model_root, hashlib.md5(src_abs.encode("utf-8")).hexdigest()
    )
    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(dest, os.path.basename(src_abs))
    if os.path.exists(dest) and verify_copy(src, dest):
        print(
            f"[WARNING]: The memory model path {dest} already exists. If it is not you want, please clear it and restart the task."
        )
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
    return dest

def _record_directory_structure(folder_path):
    record_file = os.path.join(folder_path, ".directory_record.txt")
    with open(record_file, "w") as f:
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                relative_dir = os.path.relpath(
                    os.path.join(root, dir_name), folder_path
                )
                f.write(f"dir:{relative_dir}\n")
            for file_name in files:
                if file_name != ".directory_record.txt":
                    relative_file = os.path.relpath(
                        os.path.join(root, file_name), folder_path
                    )
                    f.write(f"file:{relative_file}\n")
    return record_file

def _check_directory_structure(folder_path, record_file):
    if not os.path.exists(record_file):
        return False
    existing_entries = set()
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
            existing_entries.add(f"dir:{relative_dir}")
        for file_name in files:
            if file_name != ".directory_record.txt":
                relative_file = os.path.relpath(
                    os.path.join(root, file_name), folder_path
                )
                existing_entries.add(f"file:{relative_file}")
    with open(record_file) as f:
        recorded_entries = set(f.read().splitlines())
    return existing_entries == recorded_entries

def copy_to_local(
    src: str,
    cache_dir=None,
    filelock=".file.lock",
    verbose=False,
    always_recopy=False,
    use_shm: bool = False,
) -> str:
    local_path = copy_local_path_from_hdfs(
        src, cache_dir, filelock, verbose, always_recopy
    )
    if use_shm:
        return copy_to_shm(local_path)
    return local_path

def copy_local_path_from_hdfs(
    src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False
) -> str:
    from filelock import FileLock

    assert src[-1] != "/", (
        f"Make sure the last char in src is not / because it will cause error. Got {src}"
    )
    if is_non_local(src):
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        filelock = md5_encode(src) + ".lock"
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if always_recopy and os.path.exists(local_path):
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path, ignore_errors=True)
                else:
                    os.remove(local_path)
            if not os.path.exists(local_path):
                if verbose:
                    print(f"Copy from {src} to {local_path}")
                copy(src, local_path)
                if os.path.isdir(local_path):
                    _record_directory_structure(local_path)
            elif os.path.isdir(local_path):
                record_file = os.path.join(local_path, ".directory_record.txt")
                if not _check_directory_structure(local_path, record_file):
                    if verbose:
                        print(
                            f"Recopy from {src} to {local_path} due to missing files or directories."
                        )
                    shutil.rmtree(local_path, ignore_errors=True)
                    copy(src, local_path)
                    _record_directory_structure(local_path)
        return local_path
    else:
        return src
