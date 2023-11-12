import asyncio
from dataclasses import dataclass, field
import time
from rwkv.utils import PIPELINE_ARGS
import ray
from ray.util.queue import Queue, Empty

from model_holder import ModelHolder


@ray.remote
@dataclass
class CollectorResponse:
    minimum_time_between_responses: float
    queue: Queue = field(default_factory=Queue)
    is_finished: bool = False
    last_time: float = field(default_factory=time.time)
    last_chunk: str | None = None

    def on_end(self):
        self.is_finished = True

    def get_queue(self):
        return self.queue

    async def next(self):
        if self.is_finished and self.queue.empty():
            if self.last_chunk is not None:
                res = self.last_chunk
                self.last_chunk = None
                return (res, True)

            return None

        if self.last_chunk is None:
            self.last_chunk = await self.queue.get_async()

        now = time.time()
        while now - self.last_time < self.minimum_time_between_responses:
            diff = now - self.last_time
            await asyncio.sleep(diff)
            now = time.time()

        res = ""
        while not self.queue.empty():
            try:
                new = await self.queue.get_async(block=False)
            except Empty:
                break
            res += self.last_chunk
            self.last_chunk = new

        self.last_time = now

        return (res, False)


@ray.remote
@dataclass
class Collector:
    model_holder: ModelHolder
    minimum_time_between_responses: float = field(default=0.2)

    def generate(self, prompt: str, args: PIPELINE_ARGS, max_tokens: int):
        res = CollectorResponse.remote(self.minimum_time_between_responses)
        do_start.remote(self.model_holder, res, prompt, args, max_tokens)
        return res


@ray.remote
def do_start(
    model_holder: ModelHolder,
    res: CollectorResponse,
    prompt: str,
    args: PIPELINE_ARGS,
    max_tokens: int,
):
    ray.wait([model_holder.generate.remote(res, prompt, args, max_tokens)])
    res.on_end.remote()
