from typing import List
from pydantic import BaseModel
from typing_extensions import Annotated


class Model(BaseModel):
    name: Annotated[str, "Name of model"]


class Infer(BaseModel):
    top_k: Annotated[int, "Top-k"] = 50
    top_p: Annotated[float, "Top-p"] = 0.92
    sentences: Annotated[List[str], "Sentences to inference"]
    max_token: Annotated[int, "Max new tokens"]
    temperature: Annotated[float, "Temperature"] = 0.9
    repetition_penalty: Annotated[float, "Repetition penalty"] = 1.5
    no_repeat_ngram_size: Annotated[int, "No repeat ngram size"] =3


class Result(BaseModel):
    elapsed_time: Annotated[float, "Elapsed time"]
    used_memory: Annotated[float, "Memory used"]
