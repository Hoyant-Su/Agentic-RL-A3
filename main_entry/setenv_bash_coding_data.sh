#!/usr/bin/env bash
# Resolve dataset paths in Parquet (env_kwargs.init_dir, gold_dir, pre_files.path, …)
# relative to this directory's `data/` tree. Source before training, inference, or eval.
export BASH_CODING_DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
