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
        model["model"] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
        used_memory = torch.cuda.memory_reserved() / 1024**3

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

        model["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        model["model"] = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        used_memory = sum([torch.cuda.memory_reserved(i) / 1024**3 for i in range(8)])

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

    outputs = model["model"].generate(
        inputs,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        # min_length=10,
        # max_length=50,
        temperature=0.9,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        max_new_tokens=item["max_token"],
    )

    sentences = model["tokenizer"].batch_decode(outputs, skip_special_tokens=True)

    used_memory = torch.cuda.memory_reserved() / 1024**3
    torch.cuda.empty_cache()

    return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)


@app.post("/infer-large", status_code=200)
def generate_text_large(item: Infer):
    begin_time = time.time()
    item = item.model_dump()
    logger.info(f"Generate Text ({item}) . . .")

    sentences = item["sentences"]
    inputs = model["tokenizer"](sentences, return_tensors="pt").input_ids.cuda(7)

    outputs = model["model"].generate(
        inputs,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        # min_length=10,
        # max_length=50,
        temperature=0.9,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        max_new_tokens=item["max_token"],
    )

    sentences = model["tokenizer"].batch_decode(outputs, skip_special_tokens=True)

    used_memory = sum([torch.cuda.memory_reserved(i) / 1024**3 for i in range(8)])
    torch.cuda.empty_cache()

    return Result(elapsed_time=time.time() - begin_time, used_memory=used_memory)
