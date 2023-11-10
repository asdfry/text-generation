import os
import time
import torch
import logging

from models import Infer, Result
from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM


model = {}


@asynccontextmanager
async def load_model(app: FastAPI):
    if os.environ["DTYPE"] == "fp16":
        dytpe = torch.float16
    else:
        dytpe = torch.float32

    model_path = f"mnt/pretrained-models/{os.environ['MODEL_NAME']}"
    model["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
    model["model"] = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dytpe,
        trust_remote_code=True,
    )
    yield
    model.clear()


app = FastAPI(lifespan=load_model)
logger = logging.getLogger("uvicorn")
device_count = torch.cuda.device_count()


@app.post("/infer", status_code=200)
def generate_text_large(item: Infer):
    begin_time = time.time()
    item = item.model_dump()
    logger.info(f"Generate Text . . . {item}")

    prompt = item["sentence"]
    inputs = model["tokenizer"](prompt, return_tensors="pt").to(f"cuda:{device_count-1}")

    outputs = model["model"].generate(
        **inputs,
        top_k=item["top_k"],
        top_p=item["top_p"],
        do_sample=item["do_sample"],
        min_length=item["min_length"],
        max_length=item["max_length"],
        temperature=item["temperature"],
        repetition_penalty=item["repetition_penalty"],
        no_repeat_ngram_size=item["no_repeat_ngram_size"],
    )

    sentence = model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    sentence = sentence.split("\n")[-1].strip()

    used_memory = sum([torch.cuda.memory_reserved(i) / 1024**3 for i in range(device_count)])
    torch.cuda.empty_cache()

    return Result(sentence=sentence, elapsed_time=time.time() - begin_time, used_memory=used_memory)
