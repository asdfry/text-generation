import gc
import time
import torch
import logging

from utils import check_model
from models import Model, Infer, Result
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM


share = {"last_mem": 0.0}
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

        share["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        share["model"] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
        used_memory = torch.cuda.memory_reserved() / 1024**3 - share["last_mem"]

        return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)

    else:
        raise HTTPException(status_code=422, detail="Invalid model name")


@app.post("/load-model-large", status_code=200)
def up_model_large(item: Model):
    begin_time = time.time()
    item = item.model_dump()
    model_name = item["name"]

    if check_model(model_name):
        model_path = f"pretrained-models/{model_name}"
        logger.info(f"Load Model ({item}) . . .")

        share["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        share["model"] = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        used_memory = sum([torch.cuda.memory_reserved(i) / 1024**3 for i in range(8)]) - share["last_mem"]

        return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)

    else:
        raise HTTPException(status_code=422, detail="Invalid model name")


@app.post("/unload-model", status_code=200)
def down_model(item: Model):
    begin_time = time.time()
    item = item.model_dump()
    model_name = item["name"]

    if not "model" in share:
        raise HTTPException(status_code=404, detail="Model not loaded")

    elif check_model(model_name):
        logger.info(f"Unload Model ({item}) . . .")

        share.clear()
        torch.cuda.empty_cache()
        used_memory = torch.cuda.memory_reserved() / 1024**3
        share["last_mem"] = used_memory

        return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)

    else:
        raise HTTPException(status_code=422, detail="Invalid model name")


@app.post("/unload-model-large", status_code=200)
def down_model(item: Model):
    begin_time = time.time()
    item = item.model_dump()
    model_name = item["name"]

    if not "model" in share:
        raise HTTPException(status_code=404, detail="Model not loaded")

    elif check_model(model_name):
        logger.info(f"Unload Model ({item}) . . .")

        share["model"].cpu()
        share.clear()
        gc.collect()
        torch.cuda.empty_cache()
        used_memory = sum([torch.cuda.memory_reserved(i) / 1024**3 for i in range(8)]) / 1024**3
        share["last_mem"] = used_memory

        return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)

    else:
        raise HTTPException(status_code=422, detail="Invalid model name")


@app.post("/infer", status_code=200)
def generate_text(item: Infer):
    begin_time = time.time()
    item = item.model_dump()
    logger.info(f"Generate Text ({item}) . . .")

    sentences = item["sentences"]
    inputs = share["tokenizer"](sentences, return_tensors="pt").input_ids.cuda()

    outputs = share["model"].generate(
        inputs,
        do_sample=True,
        top_k=item["top_k"],
        top_p=item["top_p"],
        min_length=item["min_length"],
        max_length=item["max_length"],
        temperature=item["temperature"],
        repetition_penalty=item["repetition_penalty"],
        no_repeat_ngram_size=item["no_repeat_ngram_size"],
    )

    sentences = share["tokenizer"].batch_decode(outputs, skip_special_tokens=True)

    used_memory = torch.cuda.memory_reserved() / 1024**3 - share["last_mem"]
    torch.cuda.empty_cache()

    return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)


@app.post("/infer-large", status_code=200)
def generate_text_large(item: Infer):
    begin_time = time.time()
    item = item.model_dump()
    logger.info(f"Generate Text ({item}) . . .")

    sentences = item["sentences"]
    inputs = share["tokenizer"](sentences, return_tensors="pt").input_ids.cuda(7)

    outputs = share["model"].generate(
        inputs,
        do_sample=True,
        top_k=item["top_k"],
        top_p=item["top_p"],
        min_length=item["min_length"],
        max_length=item["max_length"],
        temperature=item["temperature"],
        repetition_penalty=item["repetition_penalty"],
        no_repeat_ngram_size=item["no_repeat_ngram_size"],
    )

    sentences = share["tokenizer"].batch_decode(outputs, skip_special_tokens=True)

    used_memory = sum([torch.cuda.memory_reserved(i) / 1024**3 for i in range(8)]) - share["last_mem"]
    torch.cuda.empty_cache()

    return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)
