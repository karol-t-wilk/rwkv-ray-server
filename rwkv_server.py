import json
import os
from typing import Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import ray.serve as serve


os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

from collector import Collector
from model_holder import ModelHolder
from rwkv.utils import PIPELINE_ARGS

fastapi_app = FastAPI()


class GenerationBody(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1
    top_k: int = 0
    top_p: float = 0.85
    alpha_frequency: float = 0.2
    alpha_presence: float = 0.2
    alpha_decay: float = 0.996
    token_ban: list = Field(default_factory=list)
    token_stop: list = Field(default_factory=list)
    chunk_len: int = 256


@serve.deployment(num_replicas=1)
@serve.ingress(fastapi_app)
class Model:
    def __init__(self, model_path, strategy, tokenizer_path) -> None:
        self.model_holder = ModelHolder.remote(model_path, strategy, tokenizer_path)
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
            chunk_len=body.chunk_len,
        )

        res = await self.collector.generate.remote(
            body.prompt,
            args,
            body.max_tokens,
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
