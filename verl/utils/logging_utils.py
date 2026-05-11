import logging
import os
import torch

def set_basic_config(level):
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)

def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")
