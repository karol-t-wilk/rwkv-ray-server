import json
import os
from typing import Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import ray.serve as serve


os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

from collector import Collector
from model_holder import ModelHolder
from rwkv.utils import PIPELINE_ARGS

fastapi_app = FastAPI()

class GenerationBody(BaseModel):
    prompt: str
    max_token_count: int
    temperature: float | None = None
    top_k: float | None = None
    top_p: float | None = None
    alpha_frequency: float | None = None
    alpha_presence: float | None = None
    alpha_decay: float | None = None
    token_ban: list | None = None
    token_stop: list | None = None
    chunk_len: int | None = None


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
@serve.ingress(fastapi_app)
class Model:
    def __init__(self, model_path, strategy, tokenizer_path) -> None:
        self.model_holder = ModelHolder.remote(
            model_path, strategy, tokenizer_path
        )
        self.collector = Collector.remote(self.model_holder, 0.5)

    @fastapi_app.post("/generate")
    async def handle_generate(self, body: GenerationBody) -> Any:
        args = PIPELINE_ARGS(
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            alpha_decay=body.alpha_decay,
            alpha_frequency=body.alpha_frequency,
            alpha_presence=body.alpha_presence,
            token_ban=body.token_ban,
            token_stop=body.token_stop,
            chunk_len=body.chunk_len
        )

        res = await self.collector.generate.remote(
            body.prompt,
            args,
            body.max_token_count,
        )

        async def response_stream():
            while (c := await res.next.remote()) is not None:
                chunk, is_end = c
                if chunk:
                    yield json.dumps({"text": chunk, "reached_end": is_end}) + "\n"

        return StreamingResponse(response_stream(), media_type="application/json")


def app(args: dict):
    return Model.bind(
        args["model_path"], args.get("strategy"), args.get("tokenizer_path")
    )
