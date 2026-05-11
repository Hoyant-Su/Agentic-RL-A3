
from typing import List, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs

class MultiTurnSFTDataset(Dataset):

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        assert self.truncation in ["error", "left", "right"]
        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(
                parquet_file, verbose=True
            )

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while (
                isinstance(ls, (pandas.core.series.Series, numpy.ndarray))
                and len(ls) == 1
            ):
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        full_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=False
        )
        input_ids = full_tokens[0]
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for i, msg in enumerate(messages):
            prefix_messages = messages[: i + 1]
            prefix_tokens = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            prev_tokens = (
                tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                )
                if i > 0
                else None
            )
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]
            if msg["role"] == "assistant":
                loss_mask[start_pos:end_pos] = 1
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            padded_input_ids = (
                torch.ones(
                    size=(self.max_length - sequence_length,), dtype=input_ids.dtype
                )
                * pad_token_id
            )
            padded_attention_mask = torch.zeros(
                size=(self.max_length - sequence_length,), dtype=attention_mask.dtype
            )
            padded_loss_mask = torch.zeros(
                size=(self.max_length - sequence_length,), dtype=loss_mask.dtype
            )
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(
                    f"{sequence_length=} is larger than {self.max_length=}"
                )
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = position_ids * attention_mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
