import time
import torch
import logging

from utils import check_model
from models import Model, Infer, Result
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM


model = {}
logger = logging.getLogger("uvicorn")

app = FastAPI()


@app.post("/load-model", status_code=200)
def up_model(item: Model):
    begin_time = time.time()
    item = item.model_dump()
    model_name = item["name"]

    if check_model(model_name):
        model_path = f"pretrained-models/{model_name}"
        logger.info(f"Load Model ({item}) . . .")

        model["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        model["model"] = AutoModelForCausalLM.from_pretrained(model_path).cuda()
        used_memory = torch.cuda.memory_reserved() / 1024**3

        return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)

    else:
        raise HTTPException(status_code=422, detail="Invalid model name")


@app.post("/unload-model", status_code=200)
def down_model(item: Model):
    begin_time = time.time()
    item = item.model_dump()
    model_name = item["name"]

    if not "model" in model:
        raise HTTPException(status_code=404, detail="Model not loaded")

    elif check_model(model_name):
        logger.info(f"Unload Model ({item}) . . .")

        model.clear()
        torch.cuda.empty_cache()
        used_memory = torch.cuda.memory_reserved() / 1024**3

        return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)

    else:
        raise HTTPException(status_code=422, detail="Invalid model name")


@app.post("/infer", status_code=200)
def generate_text(item: Infer):
    begin_time = time.time()
    item = item.model_dump()
    logger.info(f"Generate Text ({item}) . . .")

    sentences = item["sentences"]
    inputs = model["tokenizer"](sentences, return_tensors="pt").input_ids.cuda()

    if item["greedy"]:
        outputs = model["model"].generate(inputs, max_new_tokens=item["max_token"], do_sample=False, num_beams=1)

    elif item["top_k"]:
        outputs = model["model"].generate(inputs, max_new_tokens=item["max_token"], top_k=50)

    elif item["top_p"]:
        outputs = model["model"].generate(inputs, max_new_tokens=item["max_token"], top_k=0.95)

    sentences = model["tokenizer"].batch_decode(outputs, skip_special_tokens=True)

    used_memory = torch.cuda.memory_reserved() / 1024**3
    torch.cuda.empty_cache()

    return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)
