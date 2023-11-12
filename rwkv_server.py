import json
import os
from typing import Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import ray.serve as serve


os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

from collector import Collector
from model_holder import ModelHolder
from rwkv.utils import PIPELINE_ARGS

fastapi_app = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
@serve.ingress(fastapi_app)
class Model:
    def __init__(self, model_path, strategy, tokenizer_path) -> None:
        self.model_holder = ModelHolder.remote(
            model_path, strategy, tokenizer_path
        )
        self.collector = Collector.remote(self.model_holder, 0.5)

    @fastapi_app.post("/generate")
    async def handle_generate(self) -> Any:
        args = PIPELINE_ARGS()
        res = await self.collector.generate.remote(
            "Question: How to properly formulate a prompt for the RWKV 4 World model?\nAnswer:",
            args,
            128,
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
