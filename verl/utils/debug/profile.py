import os
import torch
import torch.distributed

class Profiler:
    def __init__(self, config):
        self.config = config
        self.skip_prof = False
        self.saved = False
        self.prof = None
        self.rank = torch.distributed.get_rank()
        self._validate()
        if config.use_profile and self.rank in self.config.profile_ranks:
            print(f"[Profiler] Profiler init for rank {self.rank}")
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=max(self.config.step_start - 1, 0),
                    warmup=1 if self.config.step_start > 0 else 0,
                    active=self.config.step_end - self.config.step_start,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )

    def _validate(self):
        if self.config.use_profile:
            if self.config.profile_ranks is None:
                print("[WARNING] Profile ranks is not set, default to rank 0")
                self.config.profile_ranks = [0]
            assert self.config.step_start >= 0, (
                "[ERROR] Profile step start must be greater than 0"
            )
            assert self.config.step_end >= 0, (
                "[ERROR] Profile step end must be greater than 0"
            )
            assert self.config.step_start < self.config.step_end, (
                "[ERROR] Profile step start must be less than step end"
            )

    def check(self):
        return self.prof is not None and not self.skip_prof

    def start(self):
        if self.check():
            print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if self.check():
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()

    def save(self):
        if self.prof is not None and not self.saved:
            if not os.path.exists(self.config.save_path):
                os.makedirs(self.config.save_path)
            save_file_name = f"/prof_start_{self.config.step_start}_end_{self.config.step_end}_rank_{self.rank}.json"
            print(
                f"[Profiler] Saving trace to {self.config.save_path + save_file_name}"
            )
            self.prof.export_chrome_trace(self.config.save_path + save_file_name)
            self.skip_prof = True
            self.saved = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def stop_trace(self):
        if self.check():
            print(f"[Profiler] Trace stopped for rank {self.rank}")
            self.skip_prof = True
