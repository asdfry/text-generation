from typing import List
from pydantic import BaseModel
from typing_extensions import Annotated


class Model(BaseModel):
    name: Annotated[str, "Name of model"]


class Infer(BaseModel):
    sentences: Annotated[List[str], "List of sentences to inference"]
    max_token: Annotated[int, "Numbers of max new tokens"]
    greedy: Annotated[bool, "Whether to use greedy search"] = False
    top_k: Annotated[bool, "Whether to use top_k"] = False
    top_p: Annotated[bool, "Whether to use top_p"] = False


class Result(BaseModel):
    elapsed_time: Annotated[float, "Elapsed time"]
    used_memory: Annotated[float, "Memory used"]
