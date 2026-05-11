import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional
import psutil
import torch
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.utils.reward_score import default_compute_score

async def single_compute_score(
    evaluation_func,
    completion,
    reference,
    task,
    task_extra_info,
    executor,
    timeout=300.0,
):
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            executor,
            partial(evaluation_func, task, completion, reference, task_extra_info),
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None

async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        try:
            tasks_async = [
                single_compute_score(
                    evaluation_func, c, r, t, ei, executor, timeout=300.0
                )
                for c, r, t, ei in zip(completions, references, tasks, extra_info)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")
    for result, completion, reference, task in zip(
        results, completions, references, tasks
    ):
        if isinstance(result, Exception) or result is None:
            scores.append(0.0)
        elif isinstance(result, (int, float, bool)):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores

def run_reward_scoring(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(
                evaluation_func,
                completions,
                references,
                tasks,
                extra_info,
                num_processes,
            )
        )
    finally:
        loop.close()

class PrimeRewardManager:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        ground_truth = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)
        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(
            scores, dtype=torch.float32, device=prompt_ids.device
        )
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )
        sequences_str = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        data_sources = data.non_tensor_batch["data_source"]
        scores = self.verify(data)
        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
