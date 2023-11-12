from dataclasses import dataclass
import ray
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


@ray.remote(num_cpus=12)
class ModelHolder:
    def __init__(self, model_path, strategy, tokenizer_path) -> None:
        if strategy is None:
            strategy = "cpu fp32"
        if tokenizer_path is None:
            tokenizer_path = "rwkv_vocab_v20230424"
        model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(model, tokenizer_path)


    async def generate(self, res, prompt: str, args: PIPELINE_ARGS, max_tokens: int):
        q = await res.get_queue.remote()
        self.pipeline.generate(
            ctx=prompt, token_count=max_tokens, args=args, callback=lambda c: q.put(c)
        )
