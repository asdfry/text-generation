from pydantic import BaseModel
from typing_extensions import Annotated


class Infer(BaseModel):
    top_k: Annotated[int, "Top-k"] = 50
    top_p: Annotated[float, "Top-p"] = 0.92
    sentence: Annotated[str, "Sentence to inference"]
    do_sample: Annotated[bool, "Whether to do sample"] = True
    min_length: Annotated[int, "Minimum length of the sequence to be generated"] = 20
    max_length: Annotated[int, "Maximum length of the sequence to be generated"] = 200
    temperature: Annotated[float, "Temperature"] = 0.9
    repetition_penalty: Annotated[float, "Repetition penalty"] = 1.5
    no_repeat_ngram_size: Annotated[int, "No repeat ngram size"] = 3


class Result(BaseModel):
    sentence: Annotated[str, "Sentence that generated"]
    elapsed_time: Annotated[float, "Elapsed time"]
    used_memory: Annotated[float, "Memory used"]
